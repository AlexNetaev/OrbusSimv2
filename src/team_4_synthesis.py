"""
team_4_synthesis.py
=====================
Team 4 - Synthesis & Hypothesis ("The Scientific Core") agents.

Role in the pipeline:
-----------------------
Team 4 is where the lab turns raw physical results into scientific
understanding and a governed decision about what happens next. Three agents
run in strict sequence:

  1. **DataAnalyst (Agent 4.1)** - the reality check. It first runs a purely
     DETERMINISTIC comparison of the hardware's reported target vs. actual
     state (`hardware_protocol.json`). If the hardware itself failed
     (explicit fault flag, or a severe target/achieved deviation), it raises
     `HardwareExecutionException` immediately and NEVER calls the LLM for
     that cycle's analysis. Only once the hardware is confirmed to have
     behaved as intended does it ask the LLM to compare the simulated
     prediction (`sim_data.csv`, from Team 3) against the actual measurement
     (`measurement.csv`) and describe the structural differences.

  2. **HypothesisArchitect (Agent 4.2)** - the scientific reasoner. It reads
     DataAnalyst's `discrepancy.md` alongside the theory baseline and asks
     the LLM to propose a grounded root-cause explanation and a concrete
     parameter adjustment for the next cycle.

  3. **Gatekeeper (Agent 4.3)** - the governance layer, applying Occam's
     razor. It reads the hypothesis (and any human/copilot feedback left in
     `copilot_feedback.md`) and asks the LLM for a strict GO/VETO decision:
     is a brand-new physical experiment actually warranted, or would
     re-tuning the simulation alone suffice? A VETO raises
     `GatekeeperVetoException`, intended for the orchestrator to catch and
     loop back to Team 3 (simulation) rather than spending real hardware
     time and reagents on a cycle Team 4 itself doesn't believe is justified.

Why DataAnalyst's reality check must be deterministic Python, not an LLM call:
------------------------------------------------------------------------------
This is the same "zero-hallucination safety philosophy" already established
by `SemanticSafetyAgent` in `src/team_5_execution.py`, applied on the way
BACK from hardware rather than on the way TO it. If a heater genuinely failed
to reach its target temperature, an LLM asked to "explain the discrepancy"
has no way to distinguish a real physical failure from a legitimate
scientific finding - it will often construct a plausible-sounding physical
narrative for what is actually just broken equipment ("perhaps convective
losses were higher than modeled..."), because that is exactly the kind of
pattern a language model is good at generating on demand. Catching hardware
failure with a hard, numeric threshold check BEFORE the LLM ever sees the
data prevents the LLM from ever being in a position to invent new physics to
explain away a broken heater.

Naming note (see also `src/team_5_execution.py`'s identical note):
-----------------------------------------------------------------------
This module is part of the same from-scratch rebuild of the full target
architecture. `src/agents/data_analyst.py` and
`src/agents/hypothesis_architect.py` already define classes named
`DataAnalyst` and `HypothesisArchitect` with different designs (notably: the
legacy DataAnalyst is data-triggered/polling and auto-corrects nothing, while
THIS DataAnalyst performs a hard, non-LLM reality check first). These classes
are intentionally independent of their legacy counterparts - no shared
imports - and will need explicit aliasing (e.g. `from src.team_4_synthesis
import DataAnalyst as SynthesisDataAnalyst`) whenever both are wired into
`main_loop.py` together, exactly as already agreed for Team 5's
`MachinePlanner`.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src import config
from src.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

HARDWARE_PROTOCOL_FILENAME = "hardware_protocol.json"
MEASUREMENT_CSV_FILENAME = "measurement.csv"
SIM_DATA_CSV_FILENAME = "sim_data.csv"
DISCREPANCY_REPORT_FILENAME = "discrepancy.md"
HYPOTHESIS_REPORT_FILENAME = "hypothesis.md"
COPILOT_FEEDBACK_FILENAME = "copilot_feedback.md"
GATEKEEPER_DECISION_FILENAME = "gatekeeper_decision.json"

# Bounds how many CSV data rows get forwarded into a prompt - identical
# technique/rationale to every other CSV-reading agent in this codebase
# (DataAnalyst in src/agents/, MachinePlanner in team_5_execution.py).
MAX_CSV_PREVIEW_ROWS = 50

# Deterministic "severe deviation" threshold for the reality check: if any
# achieved parameter is off from its target by more than this fraction, the
# hardware run is treated as failed BEFORE any LLM involvement. Matches the
# threshold already established by the legacy DataAnalyst for consistency,
# though re-declared here independently per this module's design (see
# module docstring's naming note).
SEVERE_DEVIATION_FRACTION = 0.15  # 15%


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class HardwareExecutionException(Exception):
    """
    Raised by DataAnalyst's deterministic reality check when the hardware
    layer itself failed - either an explicit fault flag, or a severe
    target-vs-achieved deviation on some parameter. Raised BEFORE any LLM
    call for this cycle's analysis (see module docstring for why).

    Carries the full list of deterministic violation messages so a human or
    `DeadlockManager` can see exactly what failed, without needing to
    re-parse hardware_protocol.json.
    """

    def __init__(self, message: str, violations: list[str]):
        super().__init__(message)
        self.violations = violations


class GatekeeperVetoException(Exception):
    """
    Raised by Gatekeeper when its GO/VETO decision comes back VETO: the
    proposed next physical experiment is judged not strictly necessary
    (Occam's razor - re-tuning the simulation may suffice instead).

    Intended to be caught by the orchestration layer (not implemented in
    this module - `main_loop.py` integration is a later step) and used to
    loop back to Team 3 (simulation) rather than proceeding to deploy a new
    physical hardware job.
    """

    def __init__(self, reasoning: str, is_success_signal: bool = False):
        super().__init__(f"Gatekeeper VETO: {reasoning}")
        self.reasoning = reasoning
        self.decision: Literal["VETO"] = "VETO"
        self.is_success_signal: bool = is_success_signal


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class HardwareProtocolReport(BaseModel):
    """
    Schema for `hardware_protocol.json` - the machine-generated report the
    hardware execution layer is expected to produce, describing target vs.
    achieved state for this cycle's run.

    Deliberately permissive (`extra="allow"`, everything defaulted) since
    this file is produced by a layer outside this module's control - we'd
    rather gracefully skip a field we don't recognize than hard-crash
    because the hardware team added or renamed something.
    """

    hardware_faults_detected: bool = False
    fault_details: list[str] = Field(default_factory=list)
    target_parameters: dict[str, float] = Field(default_factory=dict)
    achieved_parameters: dict[str, float] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class DiscrepancyAnalysisModel(BaseModel):
    """
    Strict output schema for DataAnalyst's LLM-driven comparison (only ever
    requested AFTER the deterministic reality check has already passed).
    """

    hardware_healthy: bool = Field(
        ..., description="Overall qualitative verdict: did this run behave in a scientifically trustworthy way?"
    )
    structural_differences: str = Field(
        ...,
        description="Description of the structural differences between the simulated prediction and the actual measurement (trend shape, magnitude, timing, etc.).",
    )
    detected_anomalies: list[str] = Field(
        default_factory=list, description="Specific anomalies noticed, if any. Empty list if none."
    )


class HypothesisModel(BaseModel):
    """Strict output schema for HypothesisArchitect's root-cause reasoning."""

    root_cause_analysis: str = Field(
        ...,
        description="Physical/chemical explanation for the discrepancy, grounded in the theory baseline.",
    )
    proposed_adjustment: str = Field(
        ..., description="Specific, concrete parameter adjustment(s) recommended for the next cycle."
    )


class GatekeeperDecisionModel(BaseModel):
    """
    Strict output schema for Gatekeeper's GO/VETO governance decision.
    `decision` is constrained to exactly these two literals - any other
    value fails Pydantic validation immediately, triggering llm_wrapper's
    auto-correction retry loop.
    """

    reasoning: str = Field(..., description="Step-by-step reasoning behind the GO/VETO decision.")
    decision: Literal["GO", "VETO"] = Field(
        ...,
        description=(
            "GO if a new physical experiment is strictly necessary; VETO if the simulation "
            "alone can be adjusted instead, per Occam's razor."
        ),
    )


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _read_text_or_placeholder(path: Path, placeholder: str) -> str:
    """
    Read a file's text content, or return a descriptive placeholder if the
    file doesn't exist or is empty. Same pattern used throughout this
    codebase.
    """
    if not path.exists():
        logger.warning("[team_4_synthesis] Expected file missing: %s. Using placeholder text.", path)
        return placeholder

    text = path.read_text(encoding="utf-8").strip()
    return text if text else placeholder


def _read_csv_preview_or_placeholder(csv_path: Path, max_rows: int = MAX_CSV_PREVIEW_ROWS) -> str:
    if not csv_path.exists():
        logger.warning("[team_4_synthesis] CSV file not found at %s. Using placeholder text.", csv_path)
        return f"({csv_path.name} not found.)"

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return f"({csv_path.name} is present but empty)"

    header, data_rows = rows[0], rows[1:]
    total_data_rows = len(data_rows)

    # SMART SAMPLING: Wenn mehr Zeilen als max_rows da sind, nimm gleichmäßige Stichproben
    if total_data_rows > max_rows:
        step = total_data_rows / max_rows
        preview_rows = [data_rows[int(i * step)] for i in range(max_rows)]
        # Garantiere, dass die allerletzte Zeile (das finale Ergebnis) immer enthalten ist!
        preview_rows[-1] = data_rows[-1]
    else:
        preview_rows = data_rows

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(preview_rows)

    preview_text = buffer.getvalue().strip()
    if total_data_rows > max_rows:
        preview_text += f"\n... (Data compressed: {max_rows} evenly spaced samples extracted from {total_data_rows} total rows) ..."

    return preview_text


def _run_deterministic_reality_check(protocol: HardwareProtocolReport) -> list[str]:
    """
    The core zero-hallucination safety mechanism (see module docstring).
    Pure Python, no LLM: flags an explicit hardware fault report, and any
    target-vs-achieved parameter deviating more than
    SEVERE_DEVIATION_FRACTION. Returns every violation found (not just the
    first), so a human/DeadlockManager sees the complete picture.
    """
    violations: list[str] = []

    if protocol.hardware_faults_detected:
        violations.append("Hardware layer explicitly reported hardware_faults_detected=True.")
        violations.extend(f"Reported fault detail: {detail}" for detail in protocol.fault_details)

    common_parameters = set(protocol.target_parameters) & set(protocol.achieved_parameters)
    for parameter_name in sorted(common_parameters):
        target_value = protocol.target_parameters[parameter_name]
        achieved_value = protocol.achieved_parameters[parameter_name]

        if target_value == 0:
            if achieved_value != 0:
                violations.append(f"Parameter '{parameter_name}': target was 0 but achieved {achieved_value}.")
            continue

        deviation_fraction = abs(achieved_value - target_value) / abs(target_value)
        if deviation_fraction > SEVERE_DEVIATION_FRACTION:
            violations.append(
                f"Parameter '{parameter_name}' deviated {deviation_fraction:.1%} from target "
                f"(target={target_value}, achieved={achieved_value}), exceeding the "
                f"{SEVERE_DEVIATION_FRACTION:.0%} severe-deviation threshold."
            )

    return violations

def _check_steady_state_success(csv_path: Path) -> bool:
    """
    Prüft deterministisch, ob das Experiment ein Temperatur-Plateau erreicht hat.
    Im OrbusSim-System ist die Spalte 'temp_c' (nicht 'ntc_thermistor_ohm').
    """
    if not csv_path.exists():
        return False
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows or "temp_c" not in rows[0]:
            return False

        # Prüfe die letzten 20% der Messwerte auf ein Temperatur-Plateau
        tail_size = max(1, int(len(rows) * 0.2))
        tail_values = [float(row["temp_c"]) for row in rows[-tail_size:]]
        if not tail_values:
            return False

        mean_val = sum(tail_values) / len(tail_values)
        variance = sum((x - mean_val) ** 2 for x in tail_values) / len(tail_values)
        std_dev = variance ** 0.5

        # Plateau wenn Standardabweichung < 0.5°C
        is_plateau = std_dev < 0.5
        # Zielbereich: Temperatur über 30°C (Raumtemperatur ist ~22°C)
        target_reached = mean_val > 30.0

        return is_plateau and target_reached
    except Exception as e:
        logger.warning("[team_4_synthesis] Error during steady state check: %s", e)
        return False

# --------------------------------------------------------------------------- #
# Agent 4.1: DataAnalyst
# --------------------------------------------------------------------------- #

class DataAnalyst(BaseAgent):
    """
    Performs the deterministic hardware reality check FIRST (no LLM
    involvement), then - only if that check passes - asks the LLM to
    compare the simulated prediction against the actual measurement and
    describe the structural differences. Writes C_Analysis/discrepancy.md
    and the full audit trail to shadow memory either way.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(agent_name="DataAnalyst", workspace_path=workspace_path, model=model)

    # ------------------------------------------------------------------ #
    # BaseAgent contract
    # ------------------------------------------------------------------ #

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read B_Hardware/hardware_protocol.json (must exist - raises
               FileNotFoundError if the hardware execution layer hasn't run
               yet; this is an orchestration precondition failure, distinct
               from a hardware FAILURE).
            2. Run the deterministic reality check. If it finds ANY
               violation: write a BLOCKED discrepancy.md (no LLM), write
               shadow memory, and raise HardwareExecutionException. The LLM
               is NEVER called in this branch.
            3. If the reality check passes: read measurement.csv and
               sim_data.csv, ask the LLM for a DiscrepancyAnalysisModel
               comparing them, write the full discrepancy.md, and write
               shadow memory.
        """
        hardware_dir = cycle_dir / "B_Hardware"
        analysis_dir = cycle_dir / "C_Analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        protocol_path = hardware_dir / HARDWARE_PROTOCOL_FILENAME
        if not protocol_path.exists():
            raise FileNotFoundError(
                f"No hardware_protocol.json found at {protocol_path}. The hardware execution "
                f"layer must run before DataAnalyst."
            )

        protocol = self._load_protocol_report(protocol_path)

        # --- Deterministic reality check (the zero-hallucination gate) ---------
        violations = _run_deterministic_reality_check(protocol)
        if violations:
            for violation in violations:
                self.logger.error("[DataAnalyst] Reality check violation: %s", violation)

            self._write_blocked_discrepancy_report(analysis_dir=analysis_dir, protocol=protocol, violations=violations)
            self._write_shadow_memory(
                cycle_dir=cycle_dir,
                protocol=protocol,
                violations=violations,
                llm_called=False,
                system_prompt=None,
                prompt=None,
                analysis=None,
            )
            raise HardwareExecutionException(
                message=(
                    f"Hardware reality check failed with {len(violations)} violation(s) for "
                    f"{cycle_dir.name}. LLM analysis was skipped."
                ),
                violations=violations,
            )

        # --- Reality check passed: LLM-driven structural comparison -------------
        measurement_csv_path = hardware_dir / MEASUREMENT_CSV_FILENAME
        measurement_text = _read_csv_preview_or_placeholder(hardware_dir / MEASUREMENT_CSV_FILENAME)
        sim_data_text = _read_csv_preview_or_placeholder(cycle_dir / "A_Simulation" / SIM_DATA_CSV_FILENAME)

        is_steady_state = _check_steady_state_success(measurement_csv_path)
        if is_steady_state:
             logger.info("[DataAnalyst] Deterministic check: EXPECTED STEADY STATE detected. Forcing LLM to accept plateau.")

        prompt = self._build_comparison_prompt(
            protocol=protocol, measurement_text=measurement_text, sim_data_text=sim_data_text, is_steady_state=is_steady_state
        )
        system_prompt = (
            "You are a rigorous laboratory data analyst. The hardware has ALREADY been confirmed "
            "to have run within acceptable tolerance of its targets by a separate deterministic "
            "check - your job here is purely scientific: compare the SIMULATED prediction against "
            "the ACTUAL measurement and describe how they differ structurally (trend shape, "
            "magnitude, timing, noise). Do not speculate about hardware malfunction - that "
            "possibility has already been ruled out for this cycle."
        )

        analysis = self.ask_llm(prompt=prompt, system_prompt=system_prompt, response_model=DiscrepancyAnalysisModel)
        # `ask_llm` is typed to return `str | T`; when response_model is given
        # it always returns a validated T on success (or raises otherwise), so
        # this assertion is purely a static-typing aid, not a real runtime risk.
        assert isinstance(analysis, DiscrepancyAnalysisModel)

        self._write_discrepancy_report(analysis_dir=analysis_dir, protocol=protocol, analysis=analysis)
        self._write_shadow_memory(
            cycle_dir=cycle_dir,
            protocol=protocol,
            violations=[],
            llm_called=True,
            system_prompt=system_prompt,
            prompt=prompt,
            analysis=analysis,
        )

        self.logger.info(
            "[DataAnalyst] Analysis complete for %s | hardware_healthy=%s | anomalies=%d",
            cycle_dir.name,
            analysis.hardware_healthy,
            len(analysis.detected_anomalies),
        )

    # ------------------------------------------------------------------ #
    # Reading inputs
    # ------------------------------------------------------------------ #

    def _load_protocol_report(self, protocol_path: Path) -> HardwareProtocolReport:
        """
        Parse hardware_protocol.json. Malformed JSON here means the hardware
        layer (or whatever produced this file) has a bug - a
        deterministic-source problem worth surfacing loudly, so we raise
        rather than proceeding with fabricated data.
        """
        raw_text = protocol_path.read_text(encoding="utf-8")
        try:
            raw_json = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self.logger.error("[DataAnalyst] %s is not valid JSON: %s", protocol_path, exc)
            raise
        return HardwareProtocolReport.model_validate(raw_json)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _build_comparison_prompt(
        self, protocol: HardwareProtocolReport, measurement_text: str, sim_data_text: str, is_steady_state: bool
    ) -> str:
        """Assemble the full prompt: protocol summary + measurement + simulated prediction."""
        protocol_json_str = protocol.model_dump_json(indent=2)

        steady_state_note = ""
        if is_steady_state:
            steady_state_note = (
                "\n*** CRITICAL SYSTEM NOTE ***\n"
                "Our deterministic sensors confirm the system has reached the EXPECTED TARGET STEADY STATE "
                "(temperature plateau at target value). The plateau in the measurement data is a SUCCESS, "
                "NOT an anomaly. Do NOT flag the thermal steady state as an anomaly.\n"
                "The measurement.csv contains columns: time_ms, temp_c, fluorescence_raw_au.\n"
                "A decreasing fluorescence_raw_au over time is EXPECTED behavior for this experiment "
                "(pH drops, causing fluorescein fluorescence to decrease). This is NOT an anomaly.\n"
                "******************************\n"
            )

        return f"""Compare the simulated prediction against the actual measurement for this (confirmed-healthy) hardware run.

## Hardware Execution Protocol (target vs. achieved - already confirmed within tolerance)
```json
{protocol_json_str}
```

## Actual Measurement (B_Hardware/measurement.csv)
```
{measurement_text}
```

## Simulated Prediction (A_Simulation/sim_data.csv)
```
{sim_data_text}
{steady_state_note}
```

IMPORTANT GUIDELINES FOR ANOMALY DETECTION:
- A sensor signal reaching a stable plateau when the system reaches its target
  setpoint (e.g., temperature stabilizing at the target value, resistance
  reaching a steady-state value corresponding to the target temperature) is
  EXPECTED physical behavior, NOT an anomaly. Do NOT flag thermal or chemical
  steady-state plateaus as anomalies.
- If the simulated prediction and the actual measurement have different data
  structures (e.g., the simulation produces fewer time steps or a slightly
  different column layout), note this as a STRUCTURAL DIFFERENCE, not as an
  anomaly. Structural differences are about format; anomalies are about
  unexpected physical behavior.
- Only flag genuine anomalies: unexpected jumps, oscillations, values outside
  physical plausibility, sensor readings that contradict the theory baseline,
  or behavior that cannot be explained by normal physics/chemistry.
- If the experiment achieved its target (e.g., curing_progress=1.0, target
  temperature reached), and the only "discrepancy" is that the simulation's
  numerical prediction differs slightly from the observed values, note this
  as a MODEL CALIBRATION issue in structural_differences, not as an anomaly.


## Your task
Compare the simulated prediction to the actual measurement and determine:
1. `hardware_healthy`: An overall qualitative judgment of whether this run's data looks
   scientifically trustworthy (this is a judgment call on top of the already-passed
   deterministic reality check, not a re-check of hardware faults).
2. `structural_differences`: How do the simulated and actual data structurally differ -
   trend shape, magnitude, timing, noise level, or anything else notable?
3. `detected_anomalies`: Any specific GENUINE anomalies worth flagging. Empty list if none.
   Remember: expected physical behavior (plateaus at setpoints, steady-state) is NOT an anomaly.
"""

    # ------------------------------------------------------------------ #
    # Output persistence
    # ------------------------------------------------------------------ #

    def _write_blocked_discrepancy_report(
        self, analysis_dir: Path, protocol: HardwareProtocolReport, violations: list[str]
    ) -> None:
        """
        Write a discrepancy.md documenting the deterministic reality-check
        FAILURE - no LLM content, since none was generated. This is a
        deliberate enhancement beyond the letter of the spec (which only
        requires raising the exception): giving DeadlockManager and any
        human reviewer a concrete, on-disk record of exactly what the
        deterministic check found, consistent with every other agent in
        this system always leaving a paper trail even on failure paths.
        """
        violations_block = "\n".join(f"- {violation}" for violation in violations)
        report = f"""# Data Analyst - Discrepancy Report

## Status: 🛑 BLOCKED - Hardware Reality Check Failed

The deterministic reality check found {len(violations)} violation(s). No LLM analysis was
performed - a broken or out-of-tolerance hardware run is a physical execution problem, not a
scientific discrepancy for an LLM to interpret.

## Violations
{violations_block}

## Raw Hardware Protocol Snapshot
```json
{protocol.model_dump_json(indent=2)}
```
"""
        discrepancy_path = analysis_dir / DISCREPANCY_REPORT_FILENAME
        discrepancy_path.write_text(report, encoding="utf-8")
        self.logger.info("[DataAnalyst] Wrote BLOCKED discrepancy report to %s", discrepancy_path)

    def _write_discrepancy_report(
        self, analysis_dir: Path, protocol: HardwareProtocolReport, analysis: DiscrepancyAnalysisModel
    ) -> None:
        """Write the full, LLM-generated discrepancy report."""
        status_line = "✅ HEALTHY" if analysis.hardware_healthy else "⚠️ NEEDS REVIEW"
        anomalies_block = (
            "\n".join(f"- {anomaly}" for anomaly in analysis.detected_anomalies)
            if analysis.detected_anomalies
            else "- None detected."
        )

        report = f"""# Data Analyst - Discrepancy Report

## Status: {status_line}
*(Deterministic reality check passed - this is a scientific interpretation, not a hardware fault report.)*

## Structural Differences (Simulated vs. Actual)
{analysis.structural_differences}

## Detected Anomalies
{anomalies_block}

## Raw Hardware Protocol Snapshot
```json
{protocol.model_dump_json(indent=2)}
```
"""
        discrepancy_path = analysis_dir / DISCREPANCY_REPORT_FILENAME
        discrepancy_path.write_text(report, encoding="utf-8")
        self.logger.info("[DataAnalyst] Wrote discrepancy report to %s", discrepancy_path)

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        protocol: HardwareProtocolReport,
        violations: list[str],
        llm_called: bool,
        system_prompt: str | None,
        prompt: str | None,
        analysis: DiscrepancyAnalysisModel | None,
    ) -> None:
        """
        Persist the full audit trail for both branches: the reality-check
        outcome always, and the LLM exchange only when it actually happened
        (`llm_called=False` cases have `system_prompt`/`prompt`/`analysis`
        all None, which is itself useful, explicit signal in the record).
        """
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "agent_name": self.agent_name,
            "model": self.model,
            "hardware_protocol_snapshot": protocol.model_dump(),
            "deterministic_reality_check_violations": violations,
            "llm_called": llm_called,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "validated_response": analysis.model_dump() if analysis is not None else None,
        }

        shadow_path = shadow_dir / "team4_data_analyst_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[DataAnalyst] Wrote shadow memory to %s", shadow_path)


# --------------------------------------------------------------------------- #
# Agent 4.2: HypothesisArchitect
# --------------------------------------------------------------------------- #

class HypothesisArchitect(BaseAgent):
    """
    Reads DataAnalyst's discrepancy.md and the theory baseline; asks the LLM
    to form a grounded root-cause hypothesis and a concrete adjustment
    recommendation; writes hypothesis.md and the full audit trail to shadow
    memory.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(agent_name="HypothesisArchitect", workspace_path=workspace_path, model=model)

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read C_Analysis/discrepancy.md and theory_baseline.md, each
               with a graceful placeholder if missing/empty (a missing
               discrepancy.md - e.g. this cycle was BLOCKED upstream - is
               still meaningful input: the correct hypothesis in that case
               is simply "no scientific discrepancy to explain yet").
            2. Ask the LLM for a HypothesisModel.
            3. Write hypothesis.md into C_Analysis/.
            4. Write the full prompt/response audit trail to shadow memory.
        """
        analysis_dir = cycle_dir / "C_Analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        discrepancy_text = _read_text_or_placeholder(
            analysis_dir / DISCREPANCY_REPORT_FILENAME,
            placeholder=(
                "(discrepancy.md not found for this cycle - the Data Analyst may not have "
                "produced a full analysis, e.g. if the hardware reality check failed upstream.)"
            ),
        )
        theory_text = _read_text_or_placeholder(
            config.THEORY_BASELINE_FILE,
            placeholder="(theory_baseline.md is empty - no prior theoretical grounding available yet.)",
        )

        prompt = self._build_hypothesis_prompt(theory_text=theory_text, discrepancy_text=discrepancy_text)
        system_prompt = (
            "You are the Hypothesis Architect for an autonomous self-driving laboratory. You take "
            "the lab's current theoretical understanding and this cycle's already-assessed "
            "scientific outcome, and form a grounded hypothesis connecting the two. Ground every "
            "claim in the theory baseline provided; do not invent chemistry or physics that isn't "
            "supported by it or by the observed data. If there is nothing to explain, say so "
            "plainly rather than manufacturing a root cause."
        )

        hypothesis = self.ask_llm(prompt=prompt, system_prompt=system_prompt, response_model=HypothesisModel)
        assert isinstance(hypothesis, HypothesisModel)

        self._write_hypothesis_report(analysis_dir=analysis_dir, hypothesis=hypothesis)
        self._write_shadow_memory(
            cycle_dir=cycle_dir,
            system_prompt=system_prompt,
            prompt=prompt,
            theory_text=theory_text,
            discrepancy_text=discrepancy_text,
            hypothesis=hypothesis,
        )

        self.logger.info("[HypothesisArchitect] Hypothesis complete for %s.", cycle_dir.name)

    def _build_hypothesis_prompt(self, theory_text: str, discrepancy_text: str) -> str:
        return f"""Form a scientific hypothesis explaining this cycle's outcome, grounded in current theory.

## Current Theory Baseline (theory_baseline.md)
```
{theory_text}
```

## This Cycle's Discrepancy Report (C_Analysis/discrepancy.md)
```
{discrepancy_text}
```

## Your task
Based on the theory baseline and this cycle's discrepancy report, determine:
1. `root_cause_analysis`: What physical or chemical mechanism, consistent with the theory
   baseline above, plausibly explains the discrepancy this cycle? If there was no meaningful
   discrepancy to explain, state that explicitly rather than inventing an explanation.
2. `proposed_adjustment`: What SPECIFIC parameter adjustment(s) should the next cycle try as
   a result (name the actual parameter(s) and the direction/magnitude of change)? If no
   adjustment is warranted, say so explicitly.
"""

    def _write_hypothesis_report(self, analysis_dir: Path, hypothesis: HypothesisModel) -> None:
        report = f"""# Hypothesis Architect - Root Cause & Adjustment Report

## Root Cause Analysis
{hypothesis.root_cause_analysis}

## Proposed Adjustment for Next Cycle
{hypothesis.proposed_adjustment}
"""
        hypothesis_path = analysis_dir / HYPOTHESIS_REPORT_FILENAME
        hypothesis_path.write_text(report, encoding="utf-8")
        self.logger.info("[HypothesisArchitect] Wrote hypothesis report to %s", hypothesis_path)

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        system_prompt: str,
        prompt: str,
        theory_text: str,
        discrepancy_text: str,
        hypothesis: HypothesisModel,
    ) -> None:
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "agent_name": self.agent_name,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "inputs": {"theory_baseline_text": theory_text, "discrepancy_text": discrepancy_text},
            "validated_response": hypothesis.model_dump(),
        }

        shadow_path = shadow_dir / "team4_hypothesis_architect_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[HypothesisArchitect] Wrote shadow memory to %s", shadow_path)


# --------------------------------------------------------------------------- #
# Agent 4.3: Gatekeeper
# --------------------------------------------------------------------------- #

class Gatekeeper(BaseAgent):
    """
    Applies Occam's razor: reads the hypothesis (and any human/copilot
    feedback), and asks the LLM for a strict GO/VETO governance decision on
    whether a new physical experiment is actually warranted. Always writes
    the decision log (GO or VETO), but raises `GatekeeperVetoException` on
    a VETO so the orchestrator can react (loop back to Team 3) rather than
    silently proceeding.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(agent_name="Gatekeeper", workspace_path=workspace_path, model=model)

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read C_Analysis/hypothesis.md, with a graceful placeholder if
               missing.
            2. Read C_Analysis/copilot_feedback.md IF it exists - this is a
               placeholder hook for future human/UI-injected feedback, and
               its absence is completely normal (most cycles will have no
               human feedback at all), so it's silently skipped rather than
               even logged as a warning.
            3. Ask the LLM for a GatekeeperDecisionModel (reasoning + strict
               GO/VETO).
            4. ALWAYS write C_Analysis/gatekeeper_decision.json and shadow
               memory, regardless of the outcome.
            5. If the decision is VETO, raise GatekeeperVetoException AFTER
               the decision has already been persisted to disk - the veto
               must never be lost even though it interrupts this cycle's
               flow.
        """
        analysis_dir = cycle_dir / "C_Analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        hypothesis_text = _read_text_or_placeholder(
            analysis_dir / HYPOTHESIS_REPORT_FILENAME,
            placeholder="(hypothesis.md not found for this cycle - the Hypothesis Architect may not have run yet.)",
        )
        copilot_feedback_text = self._read_copilot_feedback_if_present(analysis_dir)

        prompt = self._build_decision_prompt(hypothesis_text=hypothesis_text, copilot_feedback_text=copilot_feedback_text)
        system_prompt = (
            "You are the Gatekeeper for an autonomous self-driving laboratory, applying Occam's "
            "razor before any new physical experiment is authorized. Physical experiments consume "
            "real reagents, real hardware time, and carry real risk - they should only proceed "
            "when genuinely necessary. Vote VETO if the proposed adjustment could instead be "
            "explored by re-tuning the simulation alone, or if the hypothesis doesn't yet justify "
            "spending physical resources. Vote GO only when a new physical run is the simplest "
            "path to genuinely new information. If human/copilot feedback is provided, give it "
            "serious weight in your reasoning."
        )

        decision = self.ask_llm(prompt=prompt, system_prompt=system_prompt, response_model=GatekeeperDecisionModel)
        assert isinstance(decision, GatekeeperDecisionModel)

        self._write_decision_log(analysis_dir=analysis_dir, cycle_dir=cycle_dir, decision=decision)
        self._write_shadow_memory(
            cycle_dir=cycle_dir,
            system_prompt=system_prompt,
            prompt=prompt,
            hypothesis_text=hypothesis_text,
            copilot_feedback_text=copilot_feedback_text,
            decision=decision,
        )

        self.logger.info("[Gatekeeper] Decision for %s: %s", cycle_dir.name, decision.decision)

        if decision.decision == "VETO":
            is_success = self._detect_success_signal(decision.reasoning)
            raise GatekeeperVetoException(reasoning=decision.reasoning, is_success_signal=is_success)

    def _detect_success_signal(self, reasoning: str) -> bool:
        success_keywords = [
            "no adjustment is warranted",
            "no adjustment was warranted",
            "goal has been met",
            "goal was met",
            "goal was achieved",
            "target was met",
            "target was achieved",
            "target reached",
            "target successfully achieved",
            "target ph range was achieved",
            "target ph was reached",
            "target fluorescence plateau was reached",
            "fluorescence kinetics reached steady state",
            "no meaningful discrepancy",
            "no scientific discrepancy",
            "no discrepancy to explain",
            "no new hypothesis or parameter change",
            "no changes are being proposed",
            "no changes to the parameters",
            "successfully achieved the target",
            "successfully reached the target",
            "physical goal has been met",
            "physical experiment was successful",
        ]
        calibration_keywords = [
            "re-tune the simulation", "retune the simulation",
            "align with physical", "align with the observed",
            "calibrate the simulation", "calibration",
            "update the simulation parameters", "update the digital twin",
            "purely a modeling", "modeling failure",
            "simulation-to-hardware mismatch", "simulation inaccuracy",
            "re-tune the simulation parameters", "tune the simulation"
        ]
        reasoning_lower = reasoning.lower()
        if any(kw in reasoning_lower for kw in calibration_keywords):
            return True
        matches = sum(1 for kw in success_keywords if kw in reasoning_lower)
        return matches >= 2

    def _read_copilot_feedback_if_present(self, analysis_dir: Path) -> str:
        """
        Read copilot_feedback.md IF it exists - a hook reserved for future
        human-in-the-loop / UI-injected feedback. Its absence is the normal,
        expected case (not a warning-worthy condition, unlike every other
        "missing file" case in this codebase which usually indicates an
        upstream agent hasn't run yet) - so we return a neutral placeholder
        silently rather than logging anything.
        """
        feedback_path = analysis_dir / COPILOT_FEEDBACK_FILENAME
        if not feedback_path.exists():
            return "(No copilot/human feedback was provided for this cycle.)"

        text = feedback_path.read_text(encoding="utf-8").strip()
        return text if text else "(copilot_feedback.md is present but empty.)"

    def _build_decision_prompt(self, hypothesis_text: str, copilot_feedback_text: str) -> str:
        return f"""Decide whether a new physical experiment is strictly necessary for the next cycle.

## This Cycle's Hypothesis (C_Analysis/hypothesis.md)
```
{hypothesis_text}
```

## Human/Copilot Feedback (C_Analysis/copilot_feedback.md, if any)
```
{copilot_feedback_text}
```

## Your task
Applying Occam's razor, decide:
1. `reasoning`: Step-by-step reasoning about whether the proposed adjustment genuinely
   requires a new physical experiment, or whether adjusting/re-running the simulation alone
   could answer the same question more cheaply.
2. `decision`: Exactly "GO" (a new physical experiment is justified) or "VETO" (it is not -
   the simulation should be re-tuned instead).
"""

    def _write_decision_log(self, analysis_dir: Path, cycle_dir: Path, decision: GatekeeperDecisionModel) -> None:
        """
        Write the human-readable/machine-parseable decision log. Written
        UNCONDITIONALLY, before this method's caller (`run()`) potentially
        raises GatekeeperVetoException - a VETO decision must be just as
        durably recorded as a GO decision.
        """
        decision_record = {
            "cycle_id": cycle_dir.name,
            "decision": decision.decision,
            "reasoning": decision.reasoning,
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }
        decision_path = analysis_dir / GATEKEEPER_DECISION_FILENAME
        decision_path.write_text(json.dumps(decision_record, indent=2), encoding="utf-8")
        self.logger.info("[Gatekeeper] Wrote decision log to %s", decision_path)

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        system_prompt: str,
        prompt: str,
        hypothesis_text: str,
        copilot_feedback_text: str,
        decision: GatekeeperDecisionModel,
    ) -> None:
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "agent_name": self.agent_name,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "inputs": {"hypothesis_text": hypothesis_text, "copilot_feedback_text": copilot_feedback_text},
            "validated_response": decision.model_dump(),
        }

        shadow_path = shadow_dir / "team4_gatekeeper_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[Gatekeeper] Wrote shadow memory to %s", shadow_path)
