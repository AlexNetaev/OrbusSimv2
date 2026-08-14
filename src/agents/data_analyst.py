"""
data_analyst.py
=================
Team 4.1 - Data Analyst Agent ("DataAnalyst").

Role in the pipeline:
-----------------------
This agent is the scientific bridge between the physical world and the rest
of the lab's decision-making. It runs AFTER MachinePlanner has deployed a job
to the hardware queue, and it is DATA-TRIGGERED: rather than assuming the
external hardware simulator has already finished by the time it's invoked, it
polls for the arrival of `measurement.csv` + `hardware_protocol.json` at the
very start of its own execution, then:

  1. Locates that cycle's hardware output (`measurement.csv` +
     `hardware_protocol.json`), polling for their arrival with periodic
     status logging (see `_wait_for_hardware_outputs`).
  2. Performs a fast, deterministic "reality check" in plain Python - no LLM
     needed to notice an explicit fault flag or a wildly-off parameter.
  3. Hands the raw data + that deterministic check to the LLM to produce a
     structured, human-readable scientific interpretation
     (`AnalysisResultModel`), because judging things like "does this kinetics
     curve look physically sane" is exactly the kind of fuzzy, qualitative
     judgment call an LLM is well suited for and hardcoded thresholds are not.
  4. Persists both the human-facing verdict (`C_Analysis/discrepancy.md`) and
     the full LLM exchange (`D_Shadow_Memory/data_analyst_shadow.json`) so
     later cycles/agents can review exactly what was seen and concluded,
     without needing to re-run the LLM call.

Why polling here instead of a standalone watcher module:
-------------------------------------------------------------
The external hardware simulator writes results asynchronously, on its own
schedule, directly to `cycle_dir/B_Hardware/`. A separate long-running
"watcher" process is one way to synchronize with that, but it adds a whole
extra piece of infrastructure (a process to deploy, monitor, and keep alive)
just to answer the question "have the files shown up yet?" - a question this
agent already needs to answer for itself before it can do anything useful.
Since every agent in this system is a short-lived, single-pass, synchronous
call anyway (see BaseAgent), it's simpler and more consistent with the rest
of the architecture for DataAnalyst to just wait, briefly and boundedly, as
the first thing it does - a lightweight, stateless synchronization point with
no extra moving parts.

Consistent with the rest of the system, this agent is stateless: everything
it needs comes from files in `cycle_dir` (or, defensively, the canonical
cycle path under `config.RESEARCH_CYCLES_DIR`), and everything it produces is
written straight back to disk. Nothing is retained on the instance across runs.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field

from src import config
from src.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Well-known filenames this agent expects the hardware phase to have produced.
# Centralized as constants so a future rename only needs to change one place.
# --------------------------------------------------------------------------- #
MEASUREMENT_CSV_FILENAME = "measurement.csv"
HARDWARE_PROTOCOL_FILENAME = "hardware_protocol.json"

# --------------------------------------------------------------------------- #
# Data-triggered execution: polling configuration.
# --------------------------------------------------------------------------- #
# This agent no longer assumes hardware output already exists by the time it
# runs. Instead, it POLLS for it - the external hardware simulator consumes
# experiment.json from the queue asynchronously and writes results directly
# to cycle_dir/B_Hardware/ whenever it finishes, on its own schedule. Rather
# than standing up a separate long-running watcher process/module, this
# agent simply waits (with periodic status logging) at the very start of its
# own run() - a lightweight, stateless way to synchronize with an external,
# out-of-process producer without any additional infrastructure.
POLL_INTERVAL_SECONDS = 10
DEFAULT_HARDWARE_WAIT_TIMEOUT_SECONDS = 300

# How many data rows of the raw CSV we forward into the LLM prompt. Kinetics
# data files can be long; we cap this to keep prompts small/cheap while still
# giving the model enough of the actual curve shape to reason about.
MAX_CSV_PREVIEW_ROWS = 50

# Deterministic "severe deviation" threshold: if any achieved parameter is
# off from its target by more than this fraction, we flag it BEFORE even
# asking the LLM, so an obviously-broken run can never slip through purely
# because the LLM under-weighted a fault. This is a cheap, explainable safety
# net that costs nothing (no LLM call) and never has false negatives due to
# LLM inattention.
SEVERE_DEVIATION_FRACTION = 0.15  # 15%


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class HardwareProtocolReport(BaseModel):
    """
    Schema for `hardware_protocol.json`, the machine-generated report the
    hardware phase (real carousel or mock harness) is expected to produce.

    We deliberately keep this permissive (`extra="allow"`, everything
    defaulted) rather than strict: the hardware/mock-hardware side of the
    system may evolve independently of this agent, and we would rather
    gracefully degrade (missing fields just don't contribute to the
    deterministic check) than have this agent hard-crash because the
    hardware team added or renamed a field.
    """

    hardware_faults_detected: bool = False
    fault_details: list[str] = Field(default_factory=list)

    # Target vs. achieved values for whatever parameters this experiment
    # controlled (e.g. {"temperature_C": 85.0, "stir_rpm": 300.0}). Using
    # plain dict[str, float] rather than a fixed set of named fields keeps
    # this agent agnostic to which specific experiment/parameters were run.
    target_parameters: dict[str, float] = Field(default_factory=dict)
    achieved_parameters: dict[str, float] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class AnalysisResultModel(BaseModel):
    """
    Strict output schema the LLM must populate. This is what actually gets
    written to `discrepancy.md` (in prose form) and archived verbatim (as
    JSON) in shadow memory.
    """

    hardware_healthy: bool = Field(
        ..., description="Overall verdict: did the hardware run in a healthy, trustworthy state?"
    )
    summary_of_kinetics: str = Field(
        ..., description="Plain-language scientific summary of what the measurement data shows."
    )
    detected_anomalies: list[str] = Field(
        default_factory=list,
        description="Specific anomalies/discrepancies found, if any. Empty list if none.",
    )


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

class DataAnalyst(BaseAgent):
    """
    Reads a cycle's hardware output, runs a deterministic sanity check, asks
    the LLM for a structured scientific interpretation, and writes both the
    human-readable verdict and the raw LLM exchange back to the cycle
    directory.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        # Fixed agent name per spec - not configurable by callers, since
        # "DataAnalyst" is this agent's identity for logging/shadow-memory
        # purposes system-wide.
        super().__init__(agent_name="DataAnalyst", workspace_path=workspace_path, model=model)

    # ------------------------------------------------------------------ #
    # BaseAgent contract
    # ------------------------------------------------------------------ #

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            0. (Data-triggered) Poll for measurement.csv + hardware_protocol.json
               to arrive in this cycle's hardware output directory, as the VERY
               FIRST step. If they don't arrive within the timeout, write a
               timeout notice to discrepancy.md and return cleanly (no exception,
               no LLM call - there's nothing to analyze yet).
            1. Locate measurement.csv + hardware_protocol.json for this cycle.
            2. If either is unexpectedly still missing despite step 0 returning
               True (see _handle_missing_hardware_output docstring) -> log a
               warning and write an explanatory discrepancy.md, then stop.
            3. Otherwise, parse the protocol report and run a deterministic
               fault/deviation check.
            4. Ask the LLM to produce a full AnalysisResultModel using the
               raw CSV + protocol + our deterministic findings as context.
            5. Write discrepancy.md (human-readable) and
               data_analyst_shadow.json (raw prompt/response, machine-readable).
        """
        # --- Step 6: data-triggered wait, as the VERY FIRST action this method
        # takes. We deliberately call this before even creating C_Analysis/
        # D_Shadow_Memory below, so the polling behavior is not entangled with
        # any other setup - it's a pure "has the external world caught up yet?"
        # gate at the top of run().
        outputs_arrived = self._wait_for_hardware_outputs(cycle_dir)

        analysis_dir = cycle_dir / "C_Analysis"
        shadow_memory_dir = cycle_dir / "D_Shadow_Memory"
        # Defensive: these should already exist (main_loop.create_cycle_directory
        # creates all four subdirs up front), but we don't want this agent to
        # hard-crash on a missing directory if it's ever invoked against a
        # hand-built or partially-constructed cycle_dir (e.g. in tests).
        analysis_dir.mkdir(parents=True, exist_ok=True)
        shadow_memory_dir.mkdir(parents=True, exist_ok=True)

        if not outputs_arrived:
            # Timed out waiting - write a clear, distinct timeout notice and
            # return CLEANLY (no exception). A timeout here is an expected,
            # recoverable operational state (the hardware simulator is simply
            # still working, or temporarily unavailable), not a bug in this
            # agent or a reason to abort the whole cycle pipeline - Chef will
            # still run afterward and can report "no results yet this cycle"
            # rather than the whole run_cycle_sequence() call failing outright.
            hardware_dir = self._locate_hardware_output_dir(cycle_dir)
            self._handle_hardware_output_timeout(
                analysis_dir=analysis_dir,
                hardware_dir=hardware_dir,
                timeout_seconds=DEFAULT_HARDWARE_WAIT_TIMEOUT_SECONDS,
            )
            return

        hardware_dir = self._locate_hardware_output_dir(cycle_dir)
        measurement_path = hardware_dir / MEASUREMENT_CSV_FILENAME
        protocol_path = hardware_dir / HARDWARE_PROTOCOL_FILENAME

        if not measurement_path.exists() or not protocol_path.exists():
            # Should be unreachable in practice - _wait_for_hardware_outputs()
            # only returns True once both files were observed to exist. Kept
            # as a defensive fallback (e.g. an external process could
            # theoretically delete a file in the brief window between our
            # check and this read) rather than assuming it can never happen.
            self._handle_missing_hardware_output(
                analysis_dir=analysis_dir,
                hardware_dir=hardware_dir,
                measurement_path=measurement_path,
                protocol_path=protocol_path,
            )
            return

        # --- Load hardware output -------------------------------------------
        protocol_report = self._load_protocol_report(protocol_path)
        csv_preview = self._read_csv_preview(measurement_path)

        # --- Deterministic reality check (no LLM - cheap, explainable, and
        #     immune to the LLM ever "talking itself out of" an obvious fault) ---
        deterministic_findings = self._run_deterministic_checks(protocol_report)
        for finding in deterministic_findings:
            self.logger.warning("[DataAnalyst] Deterministic check flagged: %s", finding)

        # --- LLM-driven scientific interpretation ---------------------------
        prompt = self._build_analysis_prompt(
            csv_preview=csv_preview,
            protocol_report=protocol_report,
            deterministic_findings=deterministic_findings,
        )
        system_prompt = (
            "You are a rigorous, skeptical laboratory data analyst. You review raw "
            "instrument measurements and a hardware execution protocol to judge whether "
            "an experimental run's hardware behaved in a healthy, trustworthy way, and "
            "to summarize what the measurement data physically shows. Be conservative: "
            "if the deterministic pre-check flagged any faults or severe parameter "
            "deviations, hardware_healthy should almost always be false unless you have "
            "a clear physical reason to believe the deviation was benign."
        )

        analysis_result = self.ask_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            response_model=AnalysisResultModel,
        )
        # `ask_llm` is typed to return `str | T`; when response_model is given
        # it always returns a validated T on success (or raises otherwise), so
        # this assertion is purely a static-typing aid, not a real runtime risk.
        assert isinstance(analysis_result, AnalysisResultModel)

        # --- Persist outputs --------------------------------------------------
        self._write_discrepancy_report(
            analysis_dir=analysis_dir,
            protocol_report=protocol_report,
            deterministic_findings=deterministic_findings,
            analysis_result=analysis_result,
        )
        self._write_shadow_memory(
            shadow_memory_dir=shadow_memory_dir,
            system_prompt=system_prompt,
            prompt=prompt,
            analysis_result=analysis_result,
            deterministic_findings=deterministic_findings,
        )

        self.logger.info(
            "[DataAnalyst] Analysis complete | hardware_healthy=%s | anomalies=%d",
            analysis_result.hardware_healthy,
            len(analysis_result.detected_anomalies),
        )

    # ------------------------------------------------------------------ #
    # Locating hardware output
    # ------------------------------------------------------------------ #

    def _locate_hardware_output_dir(self, cycle_dir: Path) -> Path:
        """
        Determine where this cycle's hardware output actually lives.

        Primary location: `<cycle_dir>/B_Hardware/` - this is what
        `main_loop.create_cycle_directory()` always creates, and what a
        HardwareExecutor agent running in-process would write to directly.

        Fallback: re-resolve the cycle's canonical path via
        `config.RESEARCH_CYCLES_DIR / cycle_dir.name / B_Hardware`. This
        matters because this agent may be invoked with a `cycle_dir` that
        isn't the exact same Path object main_loop constructed (e.g. a
        symlinked mount, a different working directory, or an external
        hardware carousel process that only knows the cycle NAME and writes
        via the canonical workspace path rather than whatever relative path
        was passed to this agent). Falling back to the canonical path lets us
        recover in that situation instead of falsely reporting "no hardware
        output" when it actually exists, just not at the literal Path we were
        handed.
        """
        primary = cycle_dir / "B_Hardware"
        if (primary / MEASUREMENT_CSV_FILENAME).exists() and (primary / HARDWARE_PROTOCOL_FILENAME).exists():
            return primary

        canonical = config.RESEARCH_CYCLES_DIR / cycle_dir.name / "B_Hardware"
        if canonical != primary and (canonical / MEASUREMENT_CSV_FILENAME).exists() and (
            canonical / HARDWARE_PROTOCOL_FILENAME
        ).exists():
            self.logger.info(
                "[DataAnalyst] Hardware output not found at %s; found via canonical path %s instead.",
                primary,
                canonical,
            )
            return canonical

        # Neither location has both files. Return `primary` as the
        # "best guess" location - the caller will detect the missing files
        # and report accordingly, and reporting against the expected/primary
        # path is more useful in the discrepancy note than the fallback one.
        return primary

    # ------------------------------------------------------------------ #
    # Data-triggered polling
    # ------------------------------------------------------------------ #

    def _wait_for_hardware_outputs(
        self,
        cycle_dir: Path,
        timeout_seconds: int = DEFAULT_HARDWARE_WAIT_TIMEOUT_SECONDS,
    ) -> bool:
        """
        Poll for the arrival of BOTH `measurement.csv` and
        `hardware_protocol.json` in this cycle's hardware output directory
        (checking the same primary-then-canonical locations as
        `_locate_hardware_output_dir`, on every poll - the external hardware
        simulator might not have existed at either path when we first
        checked, but could write to either one by the time it finishes).

        This is the data-triggered synchronization point between this agent
        and the out-of-process hardware simulator: rather than assuming the
        results are already there (Step 3-5 behavior) or standing up a
        separate watcher process, DataAnalyst itself waits, briefly and
        boundedly, logging its progress every `POLL_INTERVAL_SECONDS`.

        Args:
            cycle_dir: The cycle directory to watch for hardware output under.
            timeout_seconds: Maximum total time to wait before giving up.

        Returns:
            True as soon as both files are observed to exist.
            False if `timeout_seconds` elapses first.
        """
        start_time = time.monotonic()

        while True:
            hardware_dir = self._locate_hardware_output_dir(cycle_dir)
            measurement_path = hardware_dir / MEASUREMENT_CSV_FILENAME
            protocol_path = hardware_dir / HARDWARE_PROTOCOL_FILENAME

            if measurement_path.exists() and protocol_path.exists():
                elapsed = time.monotonic() - start_time
                self.logger.info(
                    "[DataAnalyst] Hardware outputs detected for %s after %.0fs.",
                    cycle_dir.name,
                    elapsed,
                )
                return True

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_seconds:
                self.logger.warning(
                    "[DataAnalyst] Timed out after %ds waiting for hardware outputs in %s "
                    "(expected at %s).",
                    timeout_seconds,
                    cycle_dir.name,
                    hardware_dir,
                )
                return False

            self.logger.info(
                "[DataAnalyst] Waiting for external hardware outputs in %s... "
                "(%.0fs elapsed / %ds timeout)",
                cycle_dir.name,
                elapsed,
                timeout_seconds,
            )

            # Never sleep past the remaining timeout budget - this keeps the
            # total wait bounded to timeout_seconds even if POLL_INTERVAL_SECONDS
            # doesn't evenly divide it, rather than overshooting by up to one
            # full poll interval.
            remaining = timeout_seconds - elapsed
            time.sleep(min(POLL_INTERVAL_SECONDS, remaining))

    def _handle_hardware_output_timeout(
        self,
        analysis_dir: Path,
        hardware_dir: Path,
        timeout_seconds: int,
    ) -> None:
        """
        Write a clear, distinct TIMEOUT notice to discrepancy.md when
        `_wait_for_hardware_outputs` gives up without ever seeing both
        files. No LLM call is made - there is nothing to analyze.

        Deliberately kept as its own method (distinct from
        `_handle_missing_hardware_output`) because the two situations,
        while superficially similar (both end up with no hardware output to
        analyze), have different operational meanings worth communicating
        differently to a human reading discrepancy.md: a TIMEOUT means we
        actively waited and the simulator simply hasn't finished (or has
        stalled) - it says nothing about whether output is still coming.
        """
        message = (
            f"Timed out after {timeout_seconds}s waiting for hardware outputs in "
            f"{hardware_dir} (polled every {POLL_INTERVAL_SECONDS}s)."
        )
        self.logger.warning("[DataAnalyst] %s", message)

        discrepancy_path = analysis_dir / "discrepancy.md"
        discrepancy_path.write_text(
            "# Data Analyst - Discrepancy Report\n\n"
            "## Status: ⏱️ TIMEOUT - Hardware Output Not Received\n\n"
            f"- Expected hardware output directory: `{hardware_dir}`\n"
            f"- Waited **{timeout_seconds} seconds** (polling every {POLL_INTERVAL_SECONDS}s) for "
            f"`{MEASUREMENT_CSV_FILENAME}` and `{HARDWARE_PROTOCOL_FILENAME}` to appear.\n\n"
            "No LLM analysis was performed because the external hardware simulator did not "
            "produce results within the timeout window. This does not necessarily indicate a "
            "failure - the hardware run may simply still be in progress. Re-running this "
            "cycle's DataAnalyst (or the next scheduled invocation) will resume the reality "
            "check normally once the output files exist.\n",
            encoding="utf-8",
        )
        self.logger.info("[DataAnalyst] Wrote timeout notice to %s", discrepancy_path)

    # ------------------------------------------------------------------ #
    # Missing-output handling
    # ------------------------------------------------------------------ #

    def _handle_missing_hardware_output(
        self,
        analysis_dir: Path,
        hardware_dir: Path,
        measurement_path: Path,
        protocol_path: Path,
    ) -> None:
        """
        No LLM call is made here - there is nothing to analyze. We log a
        warning (visible in the run's console/log output) AND write a
        discrepancy.md (visible to any human or downstream agent browsing the
        cycle directory), since a missing file matters to both audiences and
        each surface is normally consulted independently.
        """
        missing = []
        if not measurement_path.exists():
            missing.append(MEASUREMENT_CSV_FILENAME)
        if not protocol_path.exists():
            missing.append(HARDWARE_PROTOCOL_FILENAME)

        message = (
            f"Hardware output incomplete in {hardware_dir}: missing {', '.join(missing)}. "
            "Cannot perform reality check for this cycle."
        )
        self.logger.warning("[DataAnalyst] %s", message)

        discrepancy_path = analysis_dir / "discrepancy.md"
        discrepancy_path.write_text(
            "# Data Analyst - Discrepancy Report\n\n"
            "## Status: BLOCKED - Missing Hardware Output\n\n"
            f"- Expected hardware output directory: `{hardware_dir}`\n"
            f"- Missing file(s): {', '.join(f'`{name}`' for name in missing)}\n\n"
            "No LLM analysis was performed because the required hardware output "
            "files were not found. This likely means the hardware phase for this "
            "cycle has not run yet, failed before writing output, or wrote to an "
            "unexpected location.\n",
            encoding="utf-8",
        )
        self.logger.info("[DataAnalyst] Wrote discrepancy note to %s", discrepancy_path)

    # ------------------------------------------------------------------ #
    # Parsing hardware output
    # ------------------------------------------------------------------ #

    def _load_protocol_report(self, protocol_path: Path) -> HardwareProtocolReport:
        """
        Parse `hardware_protocol.json` into a HardwareProtocolReport.

        We validate defensively here (not via the LLM pipeline) because this
        file is produced by deterministic hardware/mock-hardware code, not an
        LLM - if it's malformed, that's a hardware-side bug worth surfacing
        clearly and immediately, not something we should paper over by asking
        an LLM to "fix" a corrupt sensor log.
        """
        raw_text = protocol_path.read_text(encoding="utf-8")
        try:
            raw_json = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self.logger.error(
                "[DataAnalyst] hardware_protocol.json at %s is not valid JSON: %s", protocol_path, exc
            )
            # Re-raise: a corrupt protocol file is an unrecoverable condition
            # for this agent's run() - per BaseAgent's contract, we raise
            # rather than silently proceeding with fabricated data.
            raise

        return HardwareProtocolReport.model_validate(raw_json)

    def _read_csv_preview(self, measurement_path: Path) -> str:
        """
        Read `measurement.csv` and return a bounded, re-serialized preview
        (header + up to MAX_CSV_PREVIEW_ROWS data rows) suitable for embedding
        directly in an LLM prompt.

        We round-trip through csv.reader/writer (rather than just truncating
        raw text) so that:
          - A truncated row never gets cut off mid-field, which would confuse
            the LLM into misreading column alignment.
          - We can cleanly note in the prompt exactly how many rows were
            omitted, which the LLM should know about when reasoning about the
            full experiment vs. what it's actually looking at.
        """
        with measurement_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return "(measurement.csv is present but empty)"

        header, data_rows = rows[0], rows[1:]
        total_data_rows = len(data_rows)
        preview_rows = data_rows[:MAX_CSV_PREVIEW_ROWS]

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(header)
        writer.writerows(preview_rows)

        preview_text = buffer.getvalue().strip()
        if total_data_rows > MAX_CSV_PREVIEW_ROWS:
            omitted = total_data_rows - MAX_CSV_PREVIEW_ROWS
            preview_text += f"\n... ({omitted} further data row(s) omitted for brevity) ..."

        return preview_text

    # ------------------------------------------------------------------ #
    # Deterministic reality check
    # ------------------------------------------------------------------ #

    def _run_deterministic_checks(self, protocol_report: HardwareProtocolReport) -> list[str]:
        """
        Cheap, explainable, LLM-independent checks that catch the two most
        important failure modes without spending a single token:

          1. An explicit `hardware_faults_detected` flag from the hardware
             layer itself.
          2. Any achieved parameter deviating from its target by more than
             SEVERE_DEVIATION_FRACTION.

        Returns a list of human-readable finding strings (empty if nothing
        was flagged). These findings are fed INTO the LLM prompt (so the LLM
        has them as grounding context) rather than replacing the LLM's
        judgment entirely - the LLM still produces the final structured
        verdict, but it does so already aware of anything we could determine
        with certainty in plain Python.
        """
        findings: list[str] = []

        if protocol_report.hardware_faults_detected:
            findings.append("Hardware layer explicitly reported hardware_faults_detected=True.")
            findings.extend(f"Reported fault detail: {detail}" for detail in protocol_report.fault_details)

        # Compare target vs. achieved for every parameter present in BOTH
        # dicts. Parameters only present in one side are silently skipped
        # here (not our job to guess at data-entry mismatches) - the LLM
        # will still see both raw dicts in the prompt and can flag that
        # itself as an anomaly if relevant.
        common_parameters = set(protocol_report.target_parameters) & set(protocol_report.achieved_parameters)
        for parameter_name in sorted(common_parameters):
            target_value = protocol_report.target_parameters[parameter_name]
            achieved_value = protocol_report.achieved_parameters[parameter_name]

            if target_value == 0:
                # Avoid division-by-zero; a zero target with a non-zero
                # achieved value is itself worth flagging directly.
                if achieved_value != 0:
                    findings.append(
                        f"Parameter '{parameter_name}': target was 0 but achieved {achieved_value}."
                    )
                continue

            deviation_fraction = abs(achieved_value - target_value) / abs(target_value)
            if deviation_fraction > SEVERE_DEVIATION_FRACTION:
                findings.append(
                    f"Parameter '{parameter_name}' deviated {deviation_fraction:.1%} from target "
                    f"(target={target_value}, achieved={achieved_value}), exceeding the "
                    f"{SEVERE_DEVIATION_FRACTION:.0%} severe-deviation threshold."
                )

        return findings

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _build_analysis_prompt(
        self,
        csv_preview: str,
        protocol_report: HardwareProtocolReport,
        deterministic_findings: list[str],
    ) -> str:
        """
        Assemble the full user-facing prompt: raw measurement data + the
        hardware protocol report + whatever our deterministic pre-check
        already found, then ask for the structured verdict.
        """
        protocol_json_str = protocol_report.model_dump_json(indent=2)
        findings_block = (
            "\n".join(f"- {finding}" for finding in deterministic_findings)
            if deterministic_findings
            else "(none - no explicit faults or severe parameter deviations detected)"
        )

        return f"""Analyze the following experimental hardware run and produce a structured assessment.

## Raw measurement data (measurement.csv preview)
```
{csv_preview}
```

## Hardware execution protocol (hardware_protocol.json)
```json
{protocol_json_str}
```

## Automated deterministic pre-check findings
{findings_block}

## Your task
Based on ALL of the above, determine:
1. `hardware_healthy`: Did the hardware run in a healthy, trustworthy state overall?
   Treat any deterministic pre-check finding as strong evidence toward `false` unless
   the measurement data itself clearly shows the deviation didn't compromise the result.
2. `summary_of_kinetics`: In plain scientific language, what does the measurement data
   show happened over the course of this run (trend, stability, notable transitions)?
3. `detected_anomalies`: List any specific anomalies you notice - in the data, in the
   protocol, or inherited from the deterministic pre-check findings above. Return an
   empty list if you genuinely find nothing anomalous.
"""

    # ------------------------------------------------------------------ #
    # Output persistence
    # ------------------------------------------------------------------ #

    def _write_discrepancy_report(
        self,
        analysis_dir: Path,
        protocol_report: HardwareProtocolReport,
        deterministic_findings: list[str],
        analysis_result: AnalysisResultModel,
    ) -> None:
        """
        Write the human-readable Markdown report every downstream human or
        agent will actually read. This is the "reality check" artifact the
        rest of the system (and any human monitoring the lab) relies on.
        """
        status_line = "✅ HEALTHY" if analysis_result.hardware_healthy else "⚠️ UNHEALTHY / NEEDS REVIEW"

        anomalies_block = (
            "\n".join(f"- {anomaly}" for anomaly in analysis_result.detected_anomalies)
            if analysis_result.detected_anomalies
            else "- None detected."
        )
        deterministic_block = (
            "\n".join(f"- {finding}" for finding in deterministic_findings)
            if deterministic_findings
            else "- None."
        )

        report = f"""# Data Analyst - Discrepancy Report

## Status: {status_line}

## Kinetics Summary
{analysis_result.summary_of_kinetics}

## Detected Anomalies (LLM assessment)
{anomalies_block}

## Deterministic Pre-Check Findings
{deterministic_block}

## Raw Hardware Protocol Snapshot
```json
{protocol_report.model_dump_json(indent=2)}
```
"""
        discrepancy_path = analysis_dir / "discrepancy.md"
        discrepancy_path.write_text(report, encoding="utf-8")
        self.logger.info("[DataAnalyst] Wrote discrepancy report to %s", discrepancy_path)

    def _write_shadow_memory(
        self,
        shadow_memory_dir: Path,
        system_prompt: str,
        prompt: str,
        analysis_result: AnalysisResultModel,
        deterministic_findings: list[str],
    ) -> None:
        """
        Persist the RAW LLM exchange (system prompt, user prompt, and the
        validated response) as JSON. This is deliberately kept separate from
        discrepancy.md: discrepancy.md is the polished, human-facing verdict,
        while shadow memory is the full audit trail - useful for debugging
        prompt quality, retraining/fine-tuning later, or letting a future
        agent re-examine exactly what was asked and answered without needing
        to re-run the LLM.
        """
        shadow_record = {
            "agent_name": self.agent_name,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "deterministic_findings": deterministic_findings,
            "validated_response": analysis_result.model_dump(),
        }

        shadow_path = shadow_memory_dir / "data_analyst_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[DataAnalyst] Wrote shadow memory to %s", shadow_path)
