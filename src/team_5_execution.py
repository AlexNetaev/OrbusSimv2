"""
team_5_execution.py
=====================
Team 5 - Experiment Execution ("The Compiler") agents.

Role in the pipeline:
-----------------------
Team 5 takes the lab's scientific intent (directive + theory + this cycle's
digital-twin simulation from Team 3) and "compiles" it into a concrete,
safety-checked job for the physical hardware carousel. Two agents, run in
strict sequence, do this:

  1. **MachinePlanner (Agent 5.1)** - the compiler's front end. It reads
     `A_Simulation/sim_data.csv` (Team 3's simulated prediction for this
     cycle) alongside the global directive and theory baseline, and asks the
     LLM to produce a concrete `ExperimentJobModel`: explicit parameters for
     all 4 physical carousel stations (Reagents, Process, Analytics,
     Cleanup). It writes the result to `B_Hardware/experiment.json`.

  2. **SemanticSafetyAgent (Agent 5.2)** - the compiler's back end / linter.
     It independently RE-READS `experiment.json` from disk (never trusts an
     in-memory handoff - every agent in this system is stateless and
     file-driven) and runs a purely DETERMINISTIC set of safety checks
     against `hardware_limits.yaml`, plus a strict, non-negotiable check
     that Station 4's `cleanup_routine` is present and non-empty. If
     anything fails, it raises `HardwareSafetyException` and nothing is
     deployed. If everything passes, it copies the validated
     `experiment.json` into the global `03_Hardware_Queue/` for the
     external physical carousel to pick up.

Why SemanticSafetyAgent is deterministic Python, not another LLM call:
---------------------------------------------------------------------------
This is the same design principle already established by the legacy
`MachinePlanner`'s safety clamp in `src/agents/machine_planner.py`: a safety
GATE in front of physical hardware must be something we can reason about
and guarantee completely, not something whose behavior depends on how an LLM
happened to interpret a prompt on a given call. Every check here is plain
Python comparing numbers (and one string non-emptiness check) against a
config file - no ambiguity, no LLM cost, no chance of the gate itself being
subtly miscalibrated by prompt drift. Unlike the legacy MachinePlanner's
clamp (which auto-corrects out-of-range values), this agent's failure mode
is a HARD STOP - it raises `HardwareSafetyException` rather than silently
adjusting the job, so `DeadlockManager` gets a genuine crisis to resolve
rather than a clamp decision made unilaterally and invisibly.

Naming note (why this file has its own `MachinePlanner` class):
--------------------------------------------------------------------
This module is part of a from-scratch rebuild of the full target
architecture, built alongside (not yet replacing) the simplified agents in
`src/agents/`. `src/agents/machine_planner.py` already defines a
`MachinePlanner` class with a different (flatter, non-station-based) schema.
Both classes are named `MachinePlanner` because they play the same
CONCEPTUAL role in their respective architectures - this is intentional, not
an oversight - but it means the two are NOT interchangeable and, once wiring
these into `main_loop.py` together is on the table, whichever import brings
both into the same file will need an explicit alias (e.g.
`from src.team_5_execution import MachinePlanner as CompilerMachinePlanner`).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from src import config
from src.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Filename both agents in this module read/write within cycle_dir/B_Hardware/.
EXPERIMENT_FILENAME = "experiment.json"

# How many data rows of sim_data.csv we forward into MachinePlanner's prompt.
# Mirrors DataAnalyst's identical bound on measurement.csv previews - keeps
# the prompt small/cheap while still giving the model the actual shape of
# the simulated curve to plan around.
MAX_SIM_DATA_PREVIEW_ROWS = 50


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class HardwareSafetyException(Exception):
    """
    Raised by SemanticSafetyAgent when `experiment.json` fails ANY
    deterministic safety check - a numeric parameter outside its configured
    hardware_limits.yaml range, or a missing/empty Station 4 cleanup_routine.

    Carries the full list of individual violation messages (not just the
    first one found) so a human or `DeadlockManager` can see the complete
    picture in one place, rather than fixing one issue only to immediately
    hit the next.
    """

    def __init__(self, message: str, violations: list[str]):
        super().__init__(message)
        self.violations = violations


# --------------------------------------------------------------------------- #
# Pydantic models: the 4-station experiment job
# --------------------------------------------------------------------------- #

class ReagentDose(BaseModel):
    """A single reagent's dosing instruction for Station 1."""

    reagent_name: str = Field(..., min_length=1, description="Name/identifier of the reagent to dose.")
    volume_ml: float = Field(..., ge=0, description="Volume of this reagent to dispense, in mL.")


class Station1Reagents(BaseModel):
    """Station 1: Reagents - dosing and preparation."""

    reagents: list[ReagentDose] = Field(
        ..., min_length=1, description="One or more reagent doses to prepare for this run."
    )
    preparation_notes: str = Field(
        default="", description="Any additional preparation instructions (order of addition, etc.)."
    )


class Station2Process(BaseModel):
    """Station 2: Process - mixing and heating."""

    mixing_speed_rpm: float = Field(..., ge=0, description="Mixing speed, in RPM.")
    mixing_time_s: float = Field(..., ge=0, description="Mixing duration, in seconds.")
    target_temperature_c: float = Field(..., description="Target process temperature, in degrees Celsius.")
    heating_time_s: float = Field(..., ge=0, description="Duration to hold/ramp to target temperature, in seconds.")


class Station3Analytics(BaseModel):
    """Station 3: Analytics - measurement targets."""

    measurement_type: str = Field(
        ..., min_length=1, description="What kind of measurement to take (e.g. 'temperature', 'UV-Vis')."
    )
    target_parameters: dict[str, float] = Field(
        default_factory=dict, description="Named target values this measurement should be compared against."
    )
    sampling_interval_s: float = Field(..., ge=0, description="Interval between measurement samples, in seconds.")


class Station4Cleanup(BaseModel):
    """
    Station 4: Cleanup. `cleanup_routine` is explicitly required and
    non-empty at BOTH the schema level (here) and again, independently, by
    SemanticSafetyAgent's deterministic check - see that agent's docstring
    for why the check is deliberately duplicated rather than trusted to
    schema validation alone.
    """

    cleanup_routine: str = Field(
        ..., min_length=1, description="MUST be a non-empty description of the post-run cleanup procedure."
    )
    purge_time_s: float = Field(default=0.0, ge=0, description="Purge/rinse duration, in seconds.")

    @field_validator("cleanup_routine")
    @classmethod
    def _cleanup_routine_not_blank(cls, value: str) -> str:
        """
        Reject whitespace-only strings, not just the empty string -
        `min_length=1` alone would accept `" "` as satisfying the length
        constraint, which is not a real cleanup routine.
        """
        if not value.strip():
            raise ValueError("cleanup_routine must not be empty or whitespace-only.")
        return value


class ReagentVolumesModel(BaseModel):
    base_resin_A: float = Field(..., ge=0, description="Volume of base resin A in uL.")
    photoinitiator_B: float = Field(..., ge=0, description="Volume of photoinitiator B in uL.")
    additive_C: float = Field(..., ge=0, description="Volume of additive C in uL.")

class ExperimentParametersModel(BaseModel):
    reagents: ReagentVolumesModel
    mixing_speed_rpm: float = Field(..., ge=0, description="Mixing speed in RPM.")
    mixing_time_s: float = Field(..., ge=0, description="Mixing duration in seconds.")
    target_temperature_c: float = Field(..., description="Target temperature in Celsius.")
    target_uv_intensity_mw_cm2: float = Field(..., ge=0, description="Target UV intensity in mW/cm^2.")
    exposure_time_s: float = Field(..., ge=0, description="UV exposure duration in seconds.")

class ExperimentJobModel(BaseModel):
    """
    Strict schema the LLM must populate to match the hardware simulator.
    """
    parameters: ExperimentParametersModel
    station_4_action: str = Field(
        ..., min_length=1,
        description="MUST be a non-empty action like 'CLEAN', 'UV_CURE', or 'EJECT'."
    )


# --------------------------------------------------------------------------- #
# Hardware limits parsing (self-contained - see module docstring)
# --------------------------------------------------------------------------- #

class ParameterLimit(BaseModel):
    """
    Schema for a single entry in `hardware_limits.yaml`, e.g.:

        target_temperature_c:
          min: 0
          max: 100

    Deliberately re-declared here (rather than imported from
    `src/agents/machine_planner.py`) so this module has no dependency on the
    legacy agent package - see the module docstring's naming note for why
    these two parallel architectures are being kept independent for now.
    """

    min: float | None = None
    max: float | None = None

    class Config:
        extra = "allow"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _read_text_or_placeholder(path: Path, placeholder: str) -> str:
    """
    Read a file's text content, or return a descriptive placeholder if the
    file doesn't exist or is empty. Same pattern used throughout the rest of
    this codebase (Chef, MachinePlanner, HypothesisArchitect, ...).
    """
    if not path.exists():
        logger.warning("[team_5_execution] Expected file missing: %s. Using placeholder text.", path)
        return placeholder

    text = path.read_text(encoding="utf-8").strip()
    return text if text else placeholder


def _read_csv_preview(csv_path: Path, max_rows: int = MAX_SIM_DATA_PREVIEW_ROWS) -> str:
    """
    Read a CSV file and return a bounded, re-serialized preview (header + up
    to `max_rows` data rows), noting how many rows were omitted if any.
    Identical technique to `DataAnalyst._read_csv_preview` - round-tripping
    through csv.reader/writer rather than raw text truncation avoids ever
    cutting a row off mid-field.
    """
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return f"({csv_path.name} is present but empty)"

    header, data_rows = rows[0], rows[1:]
    total_data_rows = len(data_rows)
    preview_rows = data_rows[:max_rows]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(preview_rows)

    preview_text = buffer.getvalue().strip()
    if total_data_rows > max_rows:
        omitted = total_data_rows - max_rows
        preview_text += f"\n... ({omitted} further data row(s) omitted for brevity) ..."

    return preview_text


# --------------------------------------------------------------------------- #
# Agent 5.1: MachinePlanner
# --------------------------------------------------------------------------- #

class MachinePlanner(BaseAgent):
    """
    Reads the global directive, theory baseline, and this cycle's simulated
    prediction (sim_data.csv); asks the LLM for a complete 4-station
    ExperimentJobModel; assembles the full job payload (with a Python-assigned
    job_id/cycle_id/timestamp); writes it to B_Hardware/experiment.json; and
    persists the full audit trail to shadow memory.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(agent_name="MachinePlanner", workspace_path=workspace_path, model=model)

    # ------------------------------------------------------------------ #
    # BaseAgent contract
    # ------------------------------------------------------------------ #

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read directive.md, theory_baseline.md (global), and this
               cycle's A_Simulation/sim_data.csv (Team 3's output), each
               with a graceful placeholder if missing/empty.
            2. Ask the LLM for an ExperimentJobModel (all 4 stations).
            3. Assemble the full job payload: job_id, cycle_id, created_at,
               plus the validated station data.
            4. Write it to cycle_dir/B_Hardware/experiment.json.
            5. Write the full prompt/response audit trail to shadow memory.
        """
        directive_text = _read_text_or_placeholder(
            config.DIRECTIVE_FILE, placeholder="(directive.md is empty - no directive has been set yet.)"
        )
        theory_text = _read_text_or_placeholder(
            config.THEORY_BASELINE_FILE,
            placeholder="(theory_baseline.md is empty - no prior theoretical grounding available yet.)",
        )
        sim_data_text = self._read_sim_data(cycle_dir)

        prompt = self._build_planning_prompt(
            directive_text=directive_text, theory_text=theory_text, sim_data_text=sim_data_text
        )
        system_prompt = (
            "You are the Machine Planner for an autonomous self-driving laboratory, compiling "
            "scientific intent into a concrete job for a 4-station hardware carousel: Station 1 "
            "(Reagents - dosing/preparation), Station 2 (Process - mixing/heating), Station 3 "
            "(Analytics - measurement targets), and Station 4 (Cleanup). Every station's "
            "parameters must be specific, concrete numbers or descriptions - never vague "
            "placeholders. Station 4 in particular MUST include a real, non-empty cleanup "
            "routine description; a job without one cannot be safely deployed to physical "
            "hardware."
        )

        experiment_job = self.ask_llm(prompt=prompt, system_prompt=system_prompt, response_model=ExperimentJobModel)
        # `ask_llm` is typed to return `str | T`; when response_model is given
        # it always returns a validated T on success (or raises otherwise), so
        # this assertion is purely a static-typing aid, not a real runtime risk.
        assert isinstance(experiment_job, ExperimentJobModel)

        job_payload = self._build_job_payload(cycle_dir=cycle_dir, experiment_job=experiment_job)
        self._write_experiment_json(cycle_dir=cycle_dir, job_payload=job_payload)
        self._write_shadow_memory(
            cycle_dir=cycle_dir,
            system_prompt=system_prompt,
            prompt=prompt,
            sim_data_text=sim_data_text,
            experiment_job=experiment_job,
            job_payload=job_payload,
        )

        self.logger.info(
            "[MachinePlanner] Compiled experiment job %s for %s.", job_payload["job_id"], cycle_dir.name
        )

    # ------------------------------------------------------------------ #
    # Reading inputs
    # ------------------------------------------------------------------ #

    def _read_sim_data(self, cycle_dir: Path) -> str:
        """
        Read this cycle's `A_Simulation/sim_data.csv` (Team 3's digital-twin
        prediction), or return an explicit placeholder if it's missing.

        Handled gracefully rather than raising: Team 3 and Team 5 are wired
        together at the orchestration layer (not yet done as of this
        module), and this agent should not hard-fail just because
        simulation data isn't available for a given cycle - a reasonable
        job can still be planned from the directive and theory baseline
        alone, just without the simulated prediction as additional grounding.
        """
        sim_data_path = cycle_dir / "A_Simulation" / "sim_data.csv"
        if not sim_data_path.exists():
            self.logger.warning(
                "[MachinePlanner] sim_data.csv not found at %s - planning without simulated "
                "prediction data for this cycle.",
                sim_data_path,
            )
            return "(sim_data.csv not found for this cycle - no simulated prediction available.)"

        return _read_csv_preview(sim_data_path)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _build_planning_prompt(self, directive_text: str, theory_text: str, sim_data_text: str) -> str:
        """Assemble the full prompt: directive + theory + simulated prediction."""
        return f"""Compile a concrete, 4-station experiment job for this cycle's hardware run.

## Global Directive (directive.md)
```
{directive_text}
```

## Current Theory Baseline (theory_baseline.md)
```
{theory_text}
```

## This Cycle's Simulated Prediction (A_Simulation/sim_data.csv)
```
{sim_data_text}
```

## Your task
Produce a complete `ExperimentJobModel` covering all 4 carousel stations:
- `station_1_reagents`: one or more specific reagent doses (name + volume_ml) to prepare.
- `station_2_process`: specific mixing_speed_rpm, mixing_time_s, target_temperature_c, and
  heating_time_s.
- `station_3_analytics`: the measurement_type to take, any target_parameters to compare
  against, and the sampling_interval_s.
- `station_4_cleanup`: a REAL, non-empty cleanup_routine description (this is mandatory -
  a job without one cannot be deployed to hardware), plus purge_time_s.
Ground every value in the directive, theory baseline, and simulated prediction above. Use
specific numbers throughout - never vague placeholders or ranges.
"""

    # ------------------------------------------------------------------ #
    # Job payload assembly & persistence
    # ------------------------------------------------------------------ #

    def _build_job_payload(self, cycle_dir: Path, experiment_job: ExperimentJobModel) -> dict:
        """
        Assemble the full JSON payload: a Python-assigned job_id/cycle_id/
        created_at wrapped around the LLM-validated station data. See
        `ExperimentJobModel`'s docstring for why these identifiers are
        assigned here rather than requested from the LLM.
        """
        job_id = f"{cycle_dir.name}_{uuid.uuid4().hex[:8]}"
        payload = {
            "job_id": job_id,
            "cycle_id": cycle_dir.name,
            "target_output_dir": f"02_Research_Cycles/{cycle_dir.name}/B_Hardware",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(experiment_job.model_dump())
        return payload

    def _write_experiment_json(self, cycle_dir: Path, job_payload: dict) -> None:
        """Write the compiled job to cycle_dir/B_Hardware/experiment.json."""
        hardware_dir = cycle_dir / "B_Hardware"
        hardware_dir.mkdir(parents=True, exist_ok=True)

        experiment_path = hardware_dir / EXPERIMENT_FILENAME
        experiment_path.write_text(json.dumps(job_payload, indent=2), encoding="utf-8")
        self.logger.info("[MachinePlanner] Wrote %s", experiment_path)

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        system_prompt: str,
        prompt: str,
        sim_data_text: str,
        experiment_job: ExperimentJobModel,
        job_payload: dict,
    ) -> None:
        """
        Persist the full LLM exchange and the final assembled payload. Named
        with a `team5_` prefix (rather than reusing `machine_planner_shadow.json`)
        specifically to avoid colliding with the legacy MachinePlanner's shadow
        file of the same conceptual name - see the module docstring's naming note.
        """
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "agent_name": self.agent_name,
            "component": "team_5_execution.MachinePlanner",
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "sim_data_text": sim_data_text,
            "validated_response": experiment_job.model_dump(),
            "job_payload": job_payload,
        }

        shadow_path = shadow_dir / "team5_machine_planner_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[MachinePlanner] Wrote shadow memory to %s", shadow_path)


# --------------------------------------------------------------------------- #
# Agent 5.2: SemanticSafetyAgent
# --------------------------------------------------------------------------- #

class SemanticSafetyAgent(BaseAgent):
    """
    Deterministic safety linter and deployment gate. Re-reads
    `B_Hardware/experiment.json` from disk, checks it against
    `hardware_limits.yaml` and the mandatory Station 4 cleanup_routine
    requirement using PLAIN PYTHON (no LLM calls at all - see module
    docstring), and either deploys the job to `03_Hardware_Queue/` or raises
    `HardwareSafetyException` with the complete list of violations found.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(agent_name="SemanticSafetyAgent", workspace_path=workspace_path, model=model)

    # ------------------------------------------------------------------ #
    # BaseAgent contract
    # ------------------------------------------------------------------ #

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read B_Hardware/experiment.json (must exist - raises
               FileNotFoundError if MachinePlanner hasn't run yet).
            2. Load hardware_limits.yaml (fail-safe: missing/empty ->
               proceed with no numeric limits enforced, but the
               cleanup_routine check ALWAYS runs regardless; malformed YAML
               -> raise, since a broken safety config is a different problem
               than "no limits configured yet").
            3. Run every deterministic check, collecting ALL violations
               (not stopping at the first one).
            4. If any violations were found: write shadow memory and raise
               HardwareSafetyException with the full list.
            5. If none: copy experiment.json to the global hardware queue,
               write shadow memory, and return.
        """
        hardware_dir = cycle_dir / "B_Hardware"
        experiment_path = hardware_dir / EXPERIMENT_FILENAME

        if not experiment_path.exists():
            # Orchestration precondition failure (MachinePlanner hasn't run),
            # not a safety-check failure - raised distinctly as
            # FileNotFoundError rather than HardwareSafetyException so the
            # two failure modes stay clearly distinguishable to whatever
            # catches them upstream.
            raise FileNotFoundError(
                f"No experiment.json found at {experiment_path}. MachinePlanner (Agent 5.1) "
                f"must run before SemanticSafetyAgent."
            )

        payload = self._load_experiment_payload(experiment_path)
        hardware_limits = self._load_hardware_limits(config.HARDWARE_LIMITS_FILE)

        violations: list[str] = []
        violations.extend(self._check_cleanup_routine(payload))
        violations.extend(self._check_hardware_limits(payload, hardware_limits))

        if violations:
            for violation in violations:
                self.logger.error("[SemanticSafetyAgent] Safety violation: %s", violation)
            self._write_shadow_memory(
                cycle_dir=cycle_dir,
                payload=payload,
                hardware_limits=hardware_limits,
                violations=violations,
                deployed=False,
            )
            raise HardwareSafetyException(
                message=(
                    f"Experiment job {payload.get('job_id', '<unknown>')} failed "
                    f"{len(violations)} safety check(s) and was NOT deployed."
                ),
                violations=violations,
            )

        self._deploy_to_hardware_queue(experiment_path=experiment_path, job_id=payload.get("job_id", "<unknown>"))
        self._write_shadow_memory(
            cycle_dir=cycle_dir, payload=payload, hardware_limits=hardware_limits, violations=[], deployed=True
        )
        self.logger.info(
            "[SemanticSafetyAgent] Job %s passed all safety checks and was deployed.", payload.get("job_id")
        )

    # ------------------------------------------------------------------ #
    # Reading inputs
    # ------------------------------------------------------------------ #

    def _load_experiment_payload(self, experiment_path: Path) -> dict:
        """
        Parse experiment.json. A malformed file here means MachinePlanner
        (or something else) wrote invalid JSON - a deterministic-source bug
        worth surfacing loudly, so we raise rather than proceeding with a
        partially-parsed or fabricated payload.
        """
        raw_text = experiment_path.read_text(encoding="utf-8")
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self.logger.error("[SemanticSafetyAgent] %s is not valid JSON: %s", experiment_path, exc)
            raise

    def _load_hardware_limits(self, path: Path) -> dict[str, ParameterLimit]:
        """
        Load hardware_limits.yaml. Same fail-safe philosophy as the legacy
        MachinePlanner's loader (re-implemented here for this module's
        independence - see module docstring):
            - missing/empty -> {} (log a warning, proceed with NO numeric
              limits enforced - note the cleanup_routine check is NOT
              gated on this and always runs regardless).
            - malformed YAML / not a mapping -> raise (a broken safety
              config is a real problem, distinct from "not configured yet").
            - a single malformed entry -> skip just that entry with a
              warning, keep the rest.
        """
        if not path.exists():
            self.logger.warning(
                "[SemanticSafetyAgent] hardware_limits.yaml not found at %s - proceeding with NO "
                "numeric safety limits enforced (cleanup_routine check still applies).",
                path,
            )
            return {}

        raw_text = path.read_text(encoding="utf-8")
        if not raw_text.strip():
            self.logger.warning(
                "[SemanticSafetyAgent] hardware_limits.yaml at %s is empty - proceeding with NO "
                "numeric safety limits enforced (cleanup_routine check still applies).",
                path,
            )
            return {}

        try:
            raw_data = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            self.logger.error("[SemanticSafetyAgent] hardware_limits.yaml at %s is not valid YAML: %s", path, exc)
            raise

        if raw_data is None:
            return {}
        if not isinstance(raw_data, dict):
            raise ValueError(
                f"hardware_limits.yaml at {path} must be a top-level mapping of "
                f"parameter_name -> {{min, max}}, got {type(raw_data).__name__}."
            )

        limits: dict[str, ParameterLimit] = {}
        for parameter_name, entry in raw_data.items():
            try:
                limits[parameter_name] = ParameterLimit.model_validate(entry if isinstance(entry, dict) else {})
            except ValidationError as exc:
                self.logger.warning(
                    "[SemanticSafetyAgent] Skipping malformed hardware_limits.yaml entry '%s': %s",
                    parameter_name,
                    exc,
                )

        return limits

    # ------------------------------------------------------------------ #
    # Deterministic checks
    # ------------------------------------------------------------------ #

    def _check_cleanup_routine(self, payload: dict) -> list[str]:
        """
        STRICTLY enforce that Station 4 has a non-empty cleanup_routine.
        This check is UNCONDITIONAL - it runs regardless of whether
        hardware_limits.yaml exists or has anything to say about cleanup,
        because a missing cleanup routine is a hygiene/safety issue on its
        own terms, not something a limits file could ever waive.
        """
        action = payload.get("station_4_action")
        if not action or not str(action).strip():
            return ["station_4_action is missing or empty - Station 4 MUST have an action."]
        return []

    def _check_hardware_limits(self, payload: dict, hardware_limits: dict[str, ParameterLimit]) -> list[str]:
        """
        Check every numeric parameter this agent knows how to check against
        its configured min/max in hardware_limits.yaml. A parameter with no
        corresponding entry in hardware_limits (including when the whole
        file is missing/empty) is simply not checked - see
        `_load_hardware_limits`'s fail-safe philosophy.

        Collects and returns ALL violations found, rather than stopping at
        the first, so a human/DeadlockManager sees the complete picture.
        """
        violations: list[str] = []

        station_2 = payload.get("station_2_process") or {}
        for field_name in ("mixing_speed_rpm", "mixing_time_s", "target_temperature_c", "heating_time_s"):
            violations.extend(
                self._check_single_value(field_name, station_2.get(field_name), hardware_limits.get(field_name))
            )

        station_3 = payload.get("station_3_analytics") or {}
        violations.extend(
            self._check_single_value(
                "sampling_interval_s",
                station_3.get("sampling_interval_s"),
                hardware_limits.get("sampling_interval_s"),
            )
        )

        station_1 = payload.get("station_1_reagents") or {}
        reagent_volume_limit = hardware_limits.get("reagent_volume_ml")
        for reagent in station_1.get("reagents", []) or []:
            reagent_name = reagent.get("reagent_name", "<unnamed>")
            violations.extend(
                self._check_single_value(
                    f"reagent[{reagent_name}].volume_ml", reagent.get("volume_ml"), reagent_volume_limit
                )
            )

        return violations

    def _check_single_value(self, field_label: str, value, limit: ParameterLimit | None) -> list[str]:
        """
        Check one numeric value against one ParameterLimit. Returns an empty
        list if there's nothing to check (no limit configured, or the value
        itself is missing from the payload - a missing REQUIRED field would
        already have been caught by ExperimentJobModel's own Pydantic
        validation when MachinePlanner wrote the file, so a missing value
        here is not this agent's concern to flag).
        """
        if limit is None or value is None:
            return []

        issues: list[str] = []
        if limit.min is not None and value < limit.min:
            issues.append(f"{field_label}={value} is below the minimum allowed {limit.min}.")
        if limit.max is not None and value > limit.max:
            issues.append(f"{field_label}={value} exceeds the maximum allowed {limit.max}.")
        return issues

    # ------------------------------------------------------------------ #
    # Deployment
    # ------------------------------------------------------------------ #

    def _deploy_to_hardware_queue(self, experiment_path: Path, job_id: str) -> None:
        """
        Copy the validated experiment.json into the global hardware queue
        for the external physical carousel to pick up. Copies the file
        VERBATIM (via shutil.copy2) rather than re-serializing the parsed
        dict, so the queued file is byte-identical to what was validated -
        no risk of a re-serialization step (key ordering, float formatting)
        silently producing something subtly different from what was checked.

        Warns (rather than blocking) if a previous job is still sitting
        unconsumed in the queue - same reasoning as the legacy
        MachinePlanner: this agent's job is to deploy the CURRENT cycle's
        validated job, not to manage hardware-layer backpressure.
        """
        config.HARDWARE_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        queue_path = config.HARDWARE_QUEUE_DIR / EXPERIMENT_FILENAME

        if queue_path.exists():
            self.logger.warning(
                "[SemanticSafetyAgent] Overwriting an existing, apparently-unconsumed job at %s "
                "(previous job may not have been picked up by the hardware layer yet).",
                queue_path,
            )

        shutil.copy2(experiment_path, queue_path)
        self.logger.info("[SemanticSafetyAgent] Deployed job %s to hardware queue: %s", job_id, queue_path)

    # ------------------------------------------------------------------ #
    # Shadow memory persistence
    # ------------------------------------------------------------------ #

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        payload: dict,
        hardware_limits: dict[str, ParameterLimit],
        violations: list[str],
        deployed: bool,
    ) -> None:
        """
        Persist the full safety-evaluation audit trail: the experiment
        payload that was checked, the hardware limits snapshot it was
        checked against, every violation found (empty list if none), and
        whether the job was ultimately deployed. Named with a `team5_`
        prefix for the same collision-avoidance reason as MachinePlanner's
        shadow file above.
        """
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "agent_name": self.agent_name,
            "component": "team_5_execution.SemanticSafetyAgent",
            "job_id": payload.get("job_id"),
            "experiment_payload": payload,
            "hardware_limits_snapshot": {name: limit.model_dump() for name, limit in hardware_limits.items()},
            "violations": violations,
            "deployed": deployed,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        shadow_path = shadow_dir / "team5_semantic_safety_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[SemanticSafetyAgent] Wrote shadow memory to %s", shadow_path)
