"""
machine_planner.py
====================
Team 5 - Machine Planner Agent ("MachinePlanner").

Role in the pipeline:
-----------------------
The Machine Planner runs at the START of every research cycle, before any
hardware has executed. Its job is to translate high-level intent into a
concrete, safe, machine-executable experiment:

  1. Read WHAT we're trying to achieve (`00_System/directive.md`).
  2. Read WHAT the machine is physically allowed to do
     (`00_System/hardware_limits.yaml`).
  3. Read WHAT we currently believe about the underlying science
     (`01_Knowledge_Base/theory_baseline.md`).
  4. Ask the LLM to propose a concrete `ExperimentPlanModel` - specific
     reagent volumes, mixing/curing parameters, and a Station 4 action.
  5. Run a DETERMINISTIC safety clamp against `hardware_limits.yaml` - this
     is the actual safety boundary, not a suggestion to the LLM. Any
     parameter outside its min/max is corrected in plain Python, regardless
     of how confidently the LLM proposed it.
  6. Deploy the resulting plan as `experiment.json` into the hardware queue
     (`03_Hardware_Queue/`) for the (real or mock) hardware layer to consume,
     with an audit copy kept inside the cycle directory.

Why the safety clamp is deterministic Python, not a Pydantic validator tied
to the LLM call:
-----------------------------------------------------------------------------
`hardware_limits.yaml` is loaded fresh from disk at RUN TIME and can change
between deployments/cycles (e.g. an operator tightens a temperature ceiling
after a near-miss). `ask_llm_with_validation()` in llm_wrapper.py validates
purely against a model's static JSON schema - it has no mechanism to inject
per-call external state (like today's hardware limits) into that validation.
Rather than bolt a mutable-global workaround onto shared infrastructure, this
agent does the equivalent safety check itself, in plain Python, AFTER a
structurally-valid `ExperimentPlanModel` comes back from the LLM. This keeps
the actual safety boundary in code we control completely, testable in
isolation, and immune to whatever the model does or doesn't attend to in its
prompt - the LLM's parameter choices are a proposal; the clamp is the
guarantee.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from src import config
from src.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# The literal filename the hardware layer (real or mock) is expected to poll
# for in the queue directory. Deliberately a single fixed name (not
# per-job-numbered) because this system processes cycles strictly
# sequentially - at most one experiment should ever be awaiting hardware
# execution at a time. See _deploy_to_hardware_queue() for what happens if a
# previous job is still sitting there unconsumed.
EXPERIMENT_QUEUE_FILENAME = "experiment.json"

# Names of every numeric parameter in ExperimentPlanModel that the safety
# clamp should check against hardware_limits.yaml. Centralized as a list
# (rather than re-deriving via reflection) so it's explicit and grep-able
# which fields participate in clamping.
NUMERIC_PARAMETER_NAMES: tuple[str, ...] = (
    "base_resin_A_ml",
    "photoinitiator_B_ml",
    "additive_C_ml",
    "mixing_speed_rpm",
    "mixing_time_s",
    "target_temperature_c",
    "target_uv_intensity_mw_cm2",
    "exposure_time_s",
)


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class ExperimentPlanModel(BaseModel):
    """
    Strict schema the LLM must populate: a complete, concrete recipe for one
    experimental run on the hardware carousel.

    Every numeric field uses `ge=0` as a baseline physical-sanity constraint
    (you cannot dispense -3 mL of resin) - this is a cheap, universal floor
    that costs nothing to enforce and catches nonsensical LLM output before
    it even reaches the hardware-limits clamp. The REAL, deployment-specific
    safety ceiling/floor still comes from hardware_limits.yaml, enforced
    separately by `_apply_hardware_safety_clamp()`.
    """

    base_resin_A_ml: float = Field(..., ge=0, description="Volume of base resin A to dispense, in mL.")
    photoinitiator_B_ml: float = Field(..., ge=0, description="Volume of photoinitiator B to dispense, in mL.")
    additive_C_ml: float = Field(..., ge=0, description="Volume of additive C to dispense, in mL.")
    mixing_speed_rpm: float = Field(..., ge=0, description="Mixing speed, in RPM.")
    mixing_time_s: float = Field(..., ge=0, description="Mixing duration, in seconds.")
    target_temperature_c: float = Field(..., description="Target process temperature, in degrees Celsius.")
    target_uv_intensity_mw_cm2: float = Field(
        ..., ge=0, description="Target UV curing intensity, in mW/cm^2."
    )
    exposure_time_s: float = Field(..., ge=0, description="UV exposure duration, in seconds.")
    station_4_action: str = Field(
        ...,
        description=(
            "The action Station 4 should perform this cycle (e.g. 'cure', 'eject', "
            "'hold', 'skip'). Must be one of the values permitted by hardware_limits.yaml "
            "if a restricted set is defined there."
        ),
    )


class ParameterLimit(BaseModel):
    """
    Schema for a single entry in `hardware_limits.yaml`, e.g.:

        target_temperature_c:
          min: 15
          max: 90
        station_4_action:
          allowed_values: ["cure", "eject", "hold", "skip"]

    `min`/`max` apply to numeric parameters; `allowed_values` applies to the
    categorical `station_4_action` field. A given entry may define either or
    both, and any field a deployment doesn't care to restrict can simply be
    omitted from the YAML entirely - the clamp step below only acts on
    parameters that actually have a limit defined.
    """

    min: float | None = None
    max: float | None = None
    allowed_values: list[str] | None = None

    class Config:
        extra = "allow"


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

class MachinePlanner(BaseAgent):
    """
    Reads the directive, hardware limits, and theory baseline; asks the LLM
    for a concrete experiment plan; deterministically clamps it to hardware
    safety limits; and deploys it to the hardware queue with a full audit
    trail.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        # Fixed agent name per spec - "MachinePlanner" is this agent's
        # identity for logging/shadow-memory purposes system-wide.
        super().__init__(agent_name="MachinePlanner", workspace_path=workspace_path, model=model)

    # ------------------------------------------------------------------ #
    # BaseAgent contract
    # ------------------------------------------------------------------ #

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read directive.md, hardware_limits.yaml, theory_baseline.md.
            2. Ask the LLM for an ExperimentPlanModel.
            3. Deterministically clamp the plan against hardware_limits.yaml.
            4. Assemble the full job payload (job_id, cycle_id, parameters,
               any safety corrections applied).
            5. Deploy to config.HARDWARE_QUEUE_DIR/experiment.json, with an
               audit copy at cycle_dir/B_Hardware/experiment.json.
            6. Write the full prompt/response/correction audit trail to
               cycle_dir/D_Shadow_Memory/machine_planner_shadow.json.
        """
        directive_text = self._read_text_or_placeholder(
            config.DIRECTIVE_FILE, placeholder="(directive.md is empty - no directive has been set yet.)"
        )
        theory_text = self._read_text_or_placeholder(
            config.THEORY_BASELINE_FILE,
            placeholder="(theory_baseline.md is empty - no prior theoretical grounding available yet.)",
        )
        hardware_limits = self._load_hardware_limits(config.HARDWARE_LIMITS_FILE)

        prompt = self._build_planning_prompt(
            directive_text=directive_text,
            theory_text=theory_text,
            hardware_limits=hardware_limits,
        )
        system_prompt = (
            "You are the Machine Planner for an autonomous self-driving laboratory. "
            "You translate scientific goals and current theory into a concrete, "
            "executable experiment recipe for a liquid-handling and UV-curing hardware "
            "carousel. Always propose values strictly within any hardware limits shown "
            "to you. Be specific and quantitative - never leave a parameter as a vague "
            "placeholder."
        )

        raw_plan = self.ask_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            response_model=ExperimentPlanModel,
        )
        # `ask_llm` is typed to return `str | T`; when response_model is given
        # it always returns a validated T on success (or raises otherwise), so
        # this assertion is purely a static-typing aid, not a real runtime risk.
        assert isinstance(raw_plan, ExperimentPlanModel)

        # --- Deterministic safety clamp (the REAL safety boundary) --------------
        corrected_plan, safety_corrections = self._apply_hardware_safety_clamp(raw_plan, hardware_limits)
        for correction in safety_corrections:
            self.logger.warning("[MachinePlanner] Safety clamp applied: %s", correction)

        # --- Assemble and deploy the job payload ---------------------------------
        job_payload = self._build_job_payload(
            cycle_dir=cycle_dir,
            plan=corrected_plan,
            safety_corrections=safety_corrections,
        )
        self._deploy_to_hardware_queue(job_payload)
        self._write_audit_copy(cycle_dir=cycle_dir, job_payload=job_payload)

        # --- Shadow memory ---------------------------------------------------------
        self._write_shadow_memory(
            cycle_dir=cycle_dir,
            system_prompt=system_prompt,
            prompt=prompt,
            raw_plan=raw_plan,
            corrected_plan=corrected_plan,
            safety_corrections=safety_corrections,
            job_payload=job_payload,
        )

        self.logger.info(
            "[MachinePlanner] Plan deployed | job_id=%s | station_4_action=%s | corrections=%d",
            job_payload["job_id"],
            corrected_plan.station_4_action,
            len(safety_corrections),
        )

    # ------------------------------------------------------------------ #
    # Reading global state
    # ------------------------------------------------------------------ #

    def _read_text_or_placeholder(self, path: Path, placeholder: str) -> str:
        """
        Read a file's text content, or return a descriptive placeholder if
        the file doesn't exist or is empty. Mirrors the same pattern used in
        `Chef._read_text_or_placeholder` - both agents need this for the
        exact same reason: `directive.md`/`theory_baseline.md` are guaranteed
        to exist (as empty files) by WorkspaceManager, and an explicit
        placeholder is far clearer prompt context than silent emptiness.
        """
        if not path.exists():
            self.logger.warning("[MachinePlanner] Expected file missing: %s. Using placeholder text.", path)
            return placeholder

        text = path.read_text(encoding="utf-8").strip()
        return text if text else placeholder

    def _load_hardware_limits(self, path: Path) -> dict[str, ParameterLimit]:
        """
        Load and parse `hardware_limits.yaml` into {parameter_name: ParameterLimit}.

        Fail-safe philosophy:
            - Missing file or empty content -> perfectly normal early-project
              state (no limits defined yet). Returns {} and logs a warning;
              planning proceeds WITHOUT clamping (there is nothing to clamp
              against). This is intentionally permissive because refusing to
              plan at all just because limits haven't been configured yet
              would block the lab from ever running its first cycle.
            - Present but syntactically INVALID YAML, or not a mapping at the
              top level -> this indicates a broken safety-configuration file,
              which is a fundamentally different situation from "no limits
              yet". We raise here rather than silently proceeding with zero
              enforcement, per BaseAgent's contract (raise on unrecoverable
              conditions) - a malformed safety file should stop the pipeline
              loudly, not fail open.
            - A single entry within an otherwise-valid file that doesn't
              match ParameterLimit's schema -> we skip JUST that entry (with
              a loud warning) rather than discarding every other valid limit
              in the file over one bad entry.
        """
        if not path.exists():
            self.logger.warning(
                "[MachinePlanner] hardware_limits.yaml not found at %s - proceeding with NO safety limits enforced.",
                path,
            )
            return {}

        raw_text = path.read_text(encoding="utf-8")
        if not raw_text.strip():
            self.logger.warning(
                "[MachinePlanner] hardware_limits.yaml at %s is empty - proceeding with NO safety limits enforced.",
                path,
            )
            return {}

        try:
            raw_data = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            self.logger.error("[MachinePlanner] hardware_limits.yaml at %s is not valid YAML: %s", path, exc)
            raise

        if raw_data is None:
            # A YAML file containing only comments/whitespace parses to None.
            self.logger.warning(
                "[MachinePlanner] hardware_limits.yaml at %s parsed to no data - "
                "proceeding with NO safety limits enforced.",
                path,
            )
            return {}

        if not isinstance(raw_data, dict):
            raise ValueError(
                f"hardware_limits.yaml at {path} must be a top-level mapping of "
                f"parameter_name -> {{min, max, allowed_values}}, got {type(raw_data).__name__}."
            )

        limits: dict[str, ParameterLimit] = {}
        for parameter_name, entry in raw_data.items():
            try:
                limits[parameter_name] = ParameterLimit.model_validate(entry if isinstance(entry, dict) else {})
            except ValidationError as exc:
                self.logger.warning(
                    "[MachinePlanner] Skipping malformed hardware_limits.yaml entry '%s': %s",
                    parameter_name,
                    exc,
                )

        return limits

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _build_planning_prompt(
        self,
        directive_text: str,
        theory_text: str,
        hardware_limits: dict[str, ParameterLimit],
    ) -> str:
        """
        Assemble the full prompt: directive + theory baseline + a
        human-readable rendering of the hardware limits, then ask for the
        structured experiment plan.

        The limits are shown to the LLM as useful GUIDANCE (so it has a
        reasonable shot at proposing values that need no correction at all),
        but - per this module's design note - they are NOT what actually
        guarantees safety. That guarantee comes from
        `_apply_hardware_safety_clamp()`, run unconditionally afterward.
        """
        if hardware_limits:
            limits_block = "\n".join(
                f"- {name}: " + ", ".join(
                    part for part in [
                        f"min={limit.min}" if limit.min is not None else None,
                        f"max={limit.max}" if limit.max is not None else None,
                        f"allowed_values={limit.allowed_values}" if limit.allowed_values else None,
                    ] if part is not None
                )
                for name, limit in hardware_limits.items()
            )
        else:
            limits_block = "(No hardware limits are currently defined - use conservative, physically reasonable values.)"

        return f"""Design a concrete experiment recipe for this cycle's hardware run.

## Global Directive (directive.md)
```
{directive_text}
```

## Current Theory Baseline (theory_baseline.md)
```
{theory_text}
```

## Hardware Safety Limits (hardware_limits.yaml)
{limits_block}

## Your task
Propose a complete, concrete `ExperimentPlanModel`: exact reagent volumes (base_resin_A_ml,
photoinitiator_B_ml, additive_C_ml), mixing parameters (mixing_speed_rpm, mixing_time_s),
process conditions (target_temperature_c, target_uv_intensity_mw_cm2, exposure_time_s), and
the Station 4 action (station_4_action) for this cycle. Every value must be a specific
number (or specific action string) - do not use placeholders or ranges. Stay strictly
within any hardware limits listed above.
"""

    # ------------------------------------------------------------------ #
    # Deterministic safety clamp
    # ------------------------------------------------------------------ #

    def _apply_hardware_safety_clamp(
        self,
        plan: ExperimentPlanModel,
        hardware_limits: dict[str, ParameterLimit],
    ) -> tuple[ExperimentPlanModel, list[str]]:
        """
        The actual safety boundary. For every numeric parameter with a
        defined min/max in hardware_limits.yaml, clamp the LLM's proposed
        value into range. For `station_4_action`, if a restricted
        `allowed_values` set is defined and the LLM's choice isn't in it,
        replace it with the first allowed value.

        We CLAMP (auto-correct) rather than raise/abort, because a single
        out-of-range proposal shouldn't halt the entire lab - the corrected,
        in-bounds plan is still a perfectly valid experiment to run, and the
        correction is recorded (both in the job payload and shadow memory)
        so it's fully visible to any human or downstream agent reviewing the
        cycle, rather than silently disappearing.

        Returns:
            (corrected_plan, corrections) where `corrections` is a list of
            human-readable strings describing every change made. Empty list
            means the LLM's proposal already satisfied every defined limit.
        """
        values = plan.model_dump()
        corrections: list[str] = []

        for parameter_name in NUMERIC_PARAMETER_NAMES:
            limit = hardware_limits.get(parameter_name)
            if limit is None:
                continue  # No limit defined for this parameter - nothing to clamp.

            original_value = values[parameter_name]
            clamped_value = original_value

            if limit.min is not None and clamped_value < limit.min:
                clamped_value = limit.min
            if limit.max is not None and clamped_value > limit.max:
                clamped_value = limit.max

            if clamped_value != original_value:
                corrections.append(
                    f"{parameter_name}: clamped {original_value} -> {clamped_value} "
                    f"(limit: min={limit.min}, max={limit.max})"
                )
                values[parameter_name] = clamped_value

        action_limit = hardware_limits.get("station_4_action")
        if action_limit is not None and action_limit.allowed_values:
            original_action = values["station_4_action"]
            if original_action not in action_limit.allowed_values:
                safe_default_action = action_limit.allowed_values[0]
                corrections.append(
                    f"station_4_action: '{original_action}' not in allowed values "
                    f"{action_limit.allowed_values}; replaced with '{safe_default_action}'"
                )
                values["station_4_action"] = safe_default_action

        # Re-validate through the model so the corrected values still satisfy
        # ExperimentPlanModel's own baseline constraints (e.g. ge=0) - clamping
        # to a YAML-defined min/max should never itself produce an invalid model,
        # but re-validating costs nothing and closes that loop defensively.
        corrected_plan = ExperimentPlanModel.model_validate(values)
        return corrected_plan, corrections

    # ------------------------------------------------------------------ #
    # Job payload construction & deployment
    # ------------------------------------------------------------------ #

    def _build_job_payload(
        self,
        cycle_dir: Path,
        plan: ExperimentPlanModel,
        safety_corrections: list[str],
    ) -> dict:
        """
        Assemble the full JSON payload the hardware layer (real or mock)
        will consume from the queue.

        `job_id` combines the cycle name with a short random suffix
        (rather than e.g. a bare incrementing counter) so job IDs stay
        unique even if a cycle is somehow re-planned - the queue consumer
        can always tell two distinct planning attempts for the same cycle
        apart. `cycle_id` is simply the cycle directory's own name
        (e.g. "Cycle_003") - this agent never needs to parse or duplicate
        main_loop.py's cycle-numbering logic; it just uses whatever
        directory it was actually invoked against as the source of truth.
        """
        job_id = f"{cycle_dir.name}_{uuid.uuid4().hex[:8]}"
        return {
            "job_id": job_id,
            "cycle_id": cycle_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parameters": plan.model_dump(),
            "safety_corrections": safety_corrections,
        }

    def _deploy_to_hardware_queue(self, job_payload: dict) -> None:
        """
        Write the job payload to `config.HARDWARE_QUEUE_DIR/experiment.json`
        - the single, fixed-name file the hardware layer polls for.

        Because this system processes cycles strictly sequentially, at most
        one experiment should ever be waiting in the queue. If a previous
        `experiment.json` is still sitting there unconsumed when we go to
        write a new one, that's worth flagging loudly (it likely means the
        hardware/mock-hardware layer hasn't picked up the last job yet, or
        crashed before consuming it) - we still proceed and overwrite it
        (this agent's job is to plan for the CURRENT cycle, not to manage
        hardware-layer backpressure), but the warning ensures the situation
        isn't silently lost.
        """
        config.HARDWARE_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        queue_path = config.HARDWARE_QUEUE_DIR / EXPERIMENT_QUEUE_FILENAME

        if queue_path.exists():
            self.logger.warning(
                "[MachinePlanner] Overwriting an existing, apparently-unconsumed job at %s "
                "(previous job may not have been picked up by the hardware layer yet).",
                queue_path,
            )

        queue_path.write_text(json.dumps(job_payload, indent=2), encoding="utf-8")
        self.logger.info("[MachinePlanner] Deployed job %s to hardware queue: %s", job_payload["job_id"], queue_path)

    def _write_audit_copy(self, cycle_dir: Path, job_payload: dict) -> None:
        """
        Write an identical copy of the deployed job to
        `cycle_dir/B_Hardware/experiment.json` purely for auditing - so
        anyone browsing THIS cycle's directory later can see exactly what
        was requested of the hardware, without needing to have caught it in
        the (single-slot, get-overwritten-next-cycle) live queue.
        """
        hardware_dir = cycle_dir / "B_Hardware"
        hardware_dir.mkdir(parents=True, exist_ok=True)

        audit_path = hardware_dir / EXPERIMENT_QUEUE_FILENAME
        audit_path.write_text(json.dumps(job_payload, indent=2), encoding="utf-8")
        self.logger.info("[MachinePlanner] Wrote audit copy to %s", audit_path)

    # ------------------------------------------------------------------ #
    # Shadow memory persistence
    # ------------------------------------------------------------------ #

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        system_prompt: str,
        prompt: str,
        raw_plan: ExperimentPlanModel,
        corrected_plan: ExperimentPlanModel,
        safety_corrections: list[str],
        job_payload: dict,
    ) -> None:
        """
        Persist the full LLM exchange, the LLM's ORIGINAL (pre-clamp)
        proposal, the corrected (post-clamp) plan, and every correction
        applied. Keeping the raw LLM proposal alongside the corrected one is
        deliberate: it lets a human (or future agent) distinguish "the model
        proposed something unsafe and we caught it" from "the model already
        proposed something safe" - valuable signal for evaluating and
        improving prompt quality over time.
        """
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "agent_name": self.agent_name,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "raw_llm_plan": raw_plan.model_dump(),
            "corrected_plan": corrected_plan.model_dump(),
            "safety_corrections": safety_corrections,
            "deployed_job_payload": job_payload,
        }

        shadow_path = shadow_dir / "machine_planner_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[MachinePlanner] Wrote shadow memory to %s", shadow_path)
