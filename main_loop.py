"""
main_loop.py
=============
Master orchestration entry point for the Self-Driving Lab - THE GRAND
ORCHESTRATION (Step 5): wires the full target blueprint together.

Responsibilities at this stage:
------------------------------------
1. Bootstrap the shared workspace (via WorkspaceManager).
2. Own the lifecycle of a single "research cycle": creating its directory
   structure on disk and sequentially invoking every blueprint team's agents
   against that cycle directory, in strict order:

        a) Team 3   (Simulation)    : PythonArchitect -> SandboxDebugger
        b) Team 5   (Execution)     : CompilerMachinePlanner -> SemanticSafetyAgent
        c) [Hardware Mocking Hook]  : wait_for_hardware()
        d) Team 4   (Synthesis)     : SynthesisDataAnalyst -> ScientificHypothesisArchitect -> Gatekeeper
        e) Team 2   (Theory)        : Theoretician -> FactChecker -> KnowledgeCurator
        f) ChefAgent (Coordinator)  : reads this cycle's analysis, updates
                                       summary.md, decides goal_achieved

3. Dynamically determine which cycle to run next by scanning
   `02_Research_Cycles/` on disk, so `main()` is safely resumable.
4. Let ChefAgent's verdict (goal_achieved) control whether the lab keeps
   running further cycles.
5. Honor the dashboard's Global E-Stop (Step 8): `run_cycle_sequence()`
   checks `config.ESTOP_FLAG_FILE` at the start of every cycle and between
   every team boundary, raising `EmergencyStopException` the moment it's
   found - `main()` catches that specifically and halts the loop entirely,
   before any of the crisis-routing logic below even runs.
6. Implement INTELLIGENT EXCEPTION ROUTING: several of the blueprint's
   agents can raise exceptions representing genuinely different KINDS of
   failure, and each kind gets handled the way that failure actually
   warrants, rather than every exception funneling through one generic
   handler:

     - `GatekeeperVetoException` is caught INSIDE `run_cycle_sequence()`
       itself (see `_run_team_4_with_veto_retry`) and triggers a direct
       loop back to Team 3 to tweak the simulation, up to
       `MAX_GATEKEEPER_RETRIES` times, before ever reaching `main()`'s
       outer handler as a hard deadlock.
     - `HallucinationException` triggers a DETERMINISTIC rollback of
       `theory_baseline.md` to its exact pre-Theoretician state (recovered
       from Theoretician's own shadow-memory snapshot - no LLM guessing
       involved), and is THEN still routed to `DeadlockManager` so the lab
       decides what to do about the cycle as a whole.
     - `SimulationFailedException`, `HardwareSafetyException`, and
       `HardwareExecutionException` are each routed to
       `DeadlockManager.resolve_deadlock()` with an EXCEPTION-SPECIFIC
       reason string (the exact violations list, or the failed script's
       captured stderr tail) so the voting agents get real, actionable
       context rather than a generic "something broke".
     - Anything else (e.g. `LLMValidationException`, or a bug surfacing as
       a bare `Exception`) falls through to the same generic
       `DeadlockManager` routing as a last resort.

Naming collisions & import aliases:
---------------------------------------
`HypothesisArchitect`, `MachinePlanner`, and `DataAnalyst` each exist as
TWO different classes in this codebase: a simplified legacy version in
`src/agents/`, and a full-blueprint version in the relevant `team_N_*.py`
module. This file imports ONLY the team-specific versions, each under an
explicit alias, exactly as agreed during the Team 4/5 integration
discussions:

    from src.team_4_synthesis import DataAnalyst as SynthesisDataAnalyst
    from src.team_4_synthesis import HypothesisArchitect as ScientificHypothesisArchitect
    from src.team_5_execution import MachinePlanner as CompilerMachinePlanner

`ChefAgent` aliases `src.agents.chef.Chef` - the blueprint's coordinator
role is fulfilled by the existing legacy Chef agent as-is; no team-specific
replacement was built for it.

Why cycles are plain numbered directories on disk:
-----------------------------------------------------
Every agent is stateless (see src/base_agent.py) and communicates purely
through files. A "cycle" is simply a directory that groups everything one
full iteration of the lab produced, so a human can `cd` into it and see
exactly what happened, agents in the NEXT cycle can read the PREVIOUS
cycle's directory as their only source of memory, and crashed/resumed runs
can detect which cycles already exist on disk and pick up from there.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
import time

from src import config
from src.agents.chef import Chef as ChefAgent
from src.deadlock_manager import DeadlockManager, DeadlockResolution
from src.llm_wrapper import LLMValidationException
from src.team_2_theory import FactChecker, HallucinationException, KnowledgeCurator, Theoretician
from src.team_3_simulation import PythonArchitect, SandboxDebugger, SimulationFailedException
from src.team_4_synthesis import DataAnalyst as SynthesisDataAnalyst
from src.team_4_synthesis import Gatekeeper, GatekeeperVetoException, HardwareExecutionException
from src.team_4_synthesis import HypothesisArchitect as ScientificHypothesisArchitect
from src.team_5_execution import HardwareSafetyException, SemanticSafetyAgent
from src.team_5_execution import MachinePlanner as CompilerMachinePlanner
from src.workspace_manager import WorkspaceManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_loop")

# --------------------------------------------------------------------------- #
# Cycle directory naming & structure
# --------------------------------------------------------------------------- #

CYCLE_DIR_NAME_TEMPLATE = "Cycle_{cycle_num:03d}"
CYCLE_DIR_NAME_PATTERN = re.compile(r"^Cycle_(\d+)$")

CYCLE_SUBDIRECTORIES: tuple[str, ...] = (
    "A_Simulation",
    "B_Hardware",
    "C_Analysis",
    "D_Shadow_Memory",
)

# Maximum number of times Team 4's veto-retry loop will send control back to
# Team 3 (re-simulate) after a GatekeeperVetoException before giving up and
# escalating to a full DeadlockManager vote. See _run_team_4_with_veto_retry.
MAX_GATEKEEPER_RETRIES = 3


# --------------------------------------------------------------------------- #
# Global E-Stop (Step 8)
# --------------------------------------------------------------------------- #

class EmergencyStopException(Exception):
    """
    Raised the moment `config.ESTOP_FLAG_FILE` is detected during a cycle -
    written by `src/ui_dashboard.py`'s Tab 1 "Global Software E-Stop" button
    (a separate process; see that module's `_trigger_estop`). Unlike every
    other exception `run_cycle_sequence()` can raise, this one is NOT a
    "crisis to resolve and continue from" - it's an explicit, deliberate
    operator command to stop. `main()` catches it as a dedicated, first-priority
    case and halts the loop entirely (never routes it to `DeadlockManager`,
    never proceeds to the next cycle) - see `main()`'s exception handling.
    """

    def __init__(self, flag_path: Path, detail: str):
        super().__init__(f"Global E-Stop flag detected at {flag_path}: {detail}")
        self.flag_path = flag_path
        self.detail = detail


def _check_estop(stage_label: str) -> None:
    """
    Check for the Global E-Stop flag and raise `EmergencyStopException`
    immediately if present. Called at the very start of `run_cycle_sequence`
    and again between every team boundary (see that function's body) -
    "ideally between team executions" per the spec - so an operator-triggered
    E-Stop is honored within one pipeline step's worth of progress, not just
    at the top of the next cycle. Also called at the top of each attempt in
    `_run_team_4_with_veto_retry`'s loop, since that loop alone can run for
    several minutes (up to 3 full simulate-and-debug passes).

    `stage_label` is purely for the log line identifying exactly where in
    the pipeline the stop was caught - useful when reviewing logs after the
    fact to see how far a cycle got before halting.
    """
    if config.ESTOP_FLAG_FILE.exists():
        detail = config.ESTOP_FLAG_FILE.read_text(encoding="utf-8").strip()
        logger.critical("[E-Stop] Flag detected at stage '%s'. Halting immediately.", stage_label)
        raise EmergencyStopException(flag_path=config.ESTOP_FLAG_FILE, detail=detail)


@dataclass(frozen=True)
class CycleContext:
    """Immutable bundle describing where a cycle's agents operate."""

    cycle_num: int
    cycle_dir: Path

    @property
    def simulation_dir(self) -> Path:
        return self.cycle_dir / "A_Simulation"

    @property
    def hardware_dir(self) -> Path:
        return self.cycle_dir / "B_Hardware"

    @property
    def analysis_dir(self) -> Path:
        return self.cycle_dir / "C_Analysis"

    @property
    def shadow_memory_dir(self) -> Path:
        return self.cycle_dir / "D_Shadow_Memory"


@dataclass(frozen=True)
class CycleResult:
    """Outcome of a completed `run_cycle_sequence()` call."""

    context: CycleContext
    goal_achieved: bool
    success_veto: bool = False


# --------------------------------------------------------------------------- #
# Cycle directory creation
# --------------------------------------------------------------------------- #

def create_cycle_directory(cycle_num: int) -> CycleContext:
    """Create (idempotently) the on-disk directory structure for one research cycle."""
    if cycle_num < 1:
        raise ValueError(f"cycle_num must be >= 1, got {cycle_num}")

    cycle_dir_name = CYCLE_DIR_NAME_TEMPLATE.format(cycle_num=cycle_num)
    cycle_dir = config.RESEARCH_CYCLES_DIR / cycle_dir_name

    already_existed = cycle_dir.exists()
    cycle_dir.mkdir(parents=True, exist_ok=True)
    for subdirectory_name in CYCLE_SUBDIRECTORIES:
        (cycle_dir / subdirectory_name).mkdir(parents=True, exist_ok=True)

    if already_existed:
        logger.info("Cycle directory already existed, verified structure: %s", cycle_dir)
    else:
        logger.info("Created new cycle directory: %s", cycle_dir)

    return CycleContext(cycle_num=cycle_num, cycle_dir=cycle_dir)


def find_next_cycle_number() -> int:
    """Scan `02_Research_Cycles/` and return (highest existing cycle) + 1, or 1 if none exist."""
    if not config.RESEARCH_CYCLES_DIR.exists():
        return 1

    highest_existing_cycle_num = 0
    for entry in config.RESEARCH_CYCLES_DIR.iterdir():
        if not entry.is_dir():
            continue
        match = CYCLE_DIR_NAME_PATTERN.match(entry.name)
        if not match:
            continue
        highest_existing_cycle_num = max(highest_existing_cycle_num, int(match.group(1)))

    next_cycle_num = highest_existing_cycle_num + 1
    logger.info(
        "Scanned %s: highest existing cycle = %d -> next cycle to run = %d",
        config.RESEARCH_CYCLES_DIR,
        highest_existing_cycle_num,
        next_cycle_num,
    )
    return next_cycle_num


# --------------------------------------------------------------------------- #
# Hardware Mocking Hook
# --------------------------------------------------------------------------- #


class HardwareTimeoutException(Exception):
    """Raised when the hardware simulator does not produce results within the timeout."""
    pass


def wait_for_hardware(cycle_dir: Path) -> None:
    """
    Hardware Mocking Hook.

    In the real system, this step would BLOCK - polling or otherwise
    waiting - for an external physical carousel to consume
    `03_Hardware_Queue/experiment.json` and asynchronously write
    `B_Hardware/{measurement.csv, hardware_protocol.json}` once the run
    finishes. No real or simulated external hardware process exists yet, so
    this function stands in for that entire step: it completes immediately
    and writes plausible DUMMY `measurement.csv` + `hardware_protocol.json`
    directly into this cycle's `B_Hardware/` directory.

    The dummy data is grounded in whatever Team 5's `experiment.json`
    actually specified (if present) rather than arbitrary numbers, so
    Team 4's `DataAnalyst` has a realistic chance of its deterministic
    reality check passing, and something coherent to compare against Team
    3's simulated prediction.
    """
    hardware_dir = cycle_dir / "B_Hardware"

    # Prüfe ob überhaupt ein Experiment geplant wurde (verhindert Hängenbleiben bei Fehlern)
    if not (hardware_dir / "experiment.json").exists():
        logger.warning("[wait_for_hardware] No experiment.json found. Skipping hardware wait.")
        return

    logger.info("[Hardware] Waiting for external simulator to complete the job...")

    timeout_seconds = config.HARDWARE_WAIT_TIMEOUT_SECONDS
    start_time = time.monotonic()

    # Warteschleife: Wir warten, bis BEIDE Dateien vom Simulator abgelegt wurden
    while True:
        if (hardware_dir / "hardware_protocol.json").exists() and \
           (hardware_dir / "measurement.csv").exists():
            break

        # Check den globalen E-Stop auch während des Wartens!
        _check_estop(f"waiting for hardware in {cycle_dir.name}")

        elapsed = time.monotonic() - start_time
        if elapsed >= timeout_seconds:
            raise HardwareTimeoutException(
                f"Hardware simulator did not produce results for {cycle_dir.name} "
                f"within {timeout_seconds}s timeout."
            )

        time.sleep(2.0) # Alle 2 Sekunden prüfen

    logger.info("[Hardware] Simulator finished! Files received.")

# --------------------------------------------------------------------------- #
# Team 4 with Gatekeeper veto-retry loop-back to Team 3
# --------------------------------------------------------------------------- #

def _write_veto_feedback(cycle_dir: Path, reasoning: str, attempt: int) -> None:
    """
    Append the Gatekeeper's VETO reasoning to a small, auditable feedback
    file inside A_Simulation/, before looping back to re-run Team 3.

    Known limitation: `PythonArchitect` (Team 3.1) does not currently READ
    this file when composing its simulation prompt - its prompt only draws
    on `directive.md` and `theory_baseline.md` (see
    `src/team_3_simulation.py`). This step's scope was `main_loop.py` and
    `src/deadlock_manager.py` only, so `team_3_simulation.py` was
    deliberately left untouched. This file is written regardless because:
    (a) it's a genuinely useful, auditable record of why each retry attempt
    happened, visible right next to the simulation script it's meant to
    influence, and (b) it's the natural, minimal hook a future one-line
    change to `PythonArchitect._build_script_prompt` could read from,
    mirroring the same "read-if-present, placeholder-if-not" pattern
    `Gatekeeper` already uses for `copilot_feedback.md`.
    """
    feedback_path = cycle_dir / "A_Simulation" / "veto_feedback.md"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    entry = f"\n## Gatekeeper VETO Feedback (retry attempt {attempt})\n{reasoning}\n"
    with feedback_path.open("a", encoding="utf-8") as f:
        f.write(entry)

    logger.info("[GatekeeperVetoRetry] Wrote veto feedback (attempt %d) to %s", attempt, feedback_path)


def _run_team_4_with_veto_retry(cycle_dir: Path) -> bool:
    """
    Run Team 4. Returns True if cycle completed (GO, success-VETO, or calibrated).
    Raises GatekeeperVetoException only for NON-success VETOs after all retries.
    """
    for attempt in range(1, MAX_GATEKEEPER_RETRIES + 1):
        _check_estop(f"Team 4 veto-retry attempt {attempt}/{MAX_GATEKEEPER_RETRIES} for {cycle_dir.name}")

        SynthesisDataAnalyst().execute(cycle_dir)
        ScientificHypothesisArchitect().execute(cycle_dir)

        try:
            Gatekeeper().execute(cycle_dir)
            if attempt > 1:
                logger.info(
                    "[GatekeeperVetoRetry] GO decision reached on retry attempt %d/%d for %s.",
                    attempt,
                    MAX_GATEKEEPER_RETRIES,
                    cycle_dir.name,
                )
            return True
        except GatekeeperVetoException as veto:
            logger.warning(
                "[GatekeeperVetoRetry] Gatekeeper VETO on attempt %d/%d for %s: %s",
                attempt,
                MAX_GATEKEEPER_RETRIES,
                cycle_dir.name,
                veto.reasoning,
            )

            if veto.is_success_signal:
                logger.info("🎯 PHYSICAL GOAL REACHED! Bypassing Deadlock/Retry loops.")
                logger.info("Gatekeeper Reasoning: %s", veto.reasoning)
                _run_calibration_phase(cycle_dir, veto.reasoning)
                return True

            if attempt >= MAX_GATEKEEPER_RETRIES:
                logger.error(
                    "[GatekeeperVetoRetry] Gatekeeper VETO persisted after %d attempts for %s - "
                    "escalating to a hard deadlock.",
                    MAX_GATEKEEPER_RETRIES,
                    cycle_dir.name,
                )
                raise

            _write_veto_feedback(cycle_dir=cycle_dir, reasoning=veto.reasoning, attempt=attempt)
            logger.info(
                "[GatekeeperVetoRetry] Looping back to Team 3 (about to attempt %d/%d) to tweak the simulation.",
                attempt + 1,
                MAX_GATEKEEPER_RETRIES,
            )
            PythonArchitect().execute(cycle_dir)
            SandboxDebugger().execute(cycle_dir)

    return True

def _run_calibration_phase(cycle_dir: Path, gatekeeper_reasoning: str) -> None:
    """
    Führt den PythonArchitect im reinen Kalibrierungs-Modus aus.
    Er soll die Simulationsparameter an die echte measurement.csv anpassen.
    """
    logger.info("--- CALIBRATION PHASE: Fitting Digital Twin to Reality ---")
    measurement_csv_path = cycle_dir / "B_Hardware" / "measurement.csv"

    calibration_prompt_override = f"""
[CALIBRATION MODE ACTIVE]
The physical experiment was a 100% success. The Gatekeeper has halted further physical runs.
Your task: Write a Python script that reads the real sensor time-series data from:
`{measurement_csv_path}`
and uses `scipy.optimize.curve_fit` (or similar parameter fitting techniques) to adjust 
the kinetic/timing parameters of the simulation so that the simulated curves match the real data.

Gatekeeper Context: {gatekeeper_reasoning}

Save the final, optimized parameters to a JSON file named `calibrated_params.json` 
in the current cycle directory. Do NOT simulate a new physical experiment.
"""
    # Wir legen die Anweisung als Datei ab, damit der Architect sie lesen kann
    directive_path = cycle_dir / "A_Simulation" / "calibration_directive.md"
    directive_path.write_text(calibration_prompt_override, encoding="utf-8")

    # Führe den Architect und Debugger im Calibration-Modus aus
    PythonArchitect().execute(cycle_dir)
    SandboxDebugger().execute(cycle_dir)
    logger.info("✅ Digital Twin successfully calibrated to physical reality.")

# --------------------------------------------------------------------------- #
# HallucinationException rollback
# --------------------------------------------------------------------------- #

def _rollback_theory_baseline(cycle_dir: Path) -> bool:
    """
    Deterministically restore `theory_baseline.md` to its EXACT
    pre-Theoretician state for this cycle, recovered from Theoretician's own
    shadow-memory snapshot (`previous_baseline_text`) - no LLM involved, so
    there is no risk of the "fix" itself hallucinating a plausible-looking
    but wrong prior state.

    Returns True if the rollback succeeded, False if the shadow-memory
    snapshot was unavailable or malformed (an extreme edge case - this
    should never happen in practice, since Theoretician always writes its
    shadow memory before FactChecker can even run, but handled defensively
    rather than raising a second exception on top of the first).
    """
    shadow_path = cycle_dir / "D_Shadow_Memory" / "team2_theoretician_shadow.json"
    if not shadow_path.exists():
        logger.error(
            "[HallucinationRollback] Cannot roll back theory_baseline.md for %s - no Theoretician "
            "shadow memory found at %s.",
            cycle_dir.name,
            shadow_path,
        )
        return False

    try:
        shadow_record = json.loads(shadow_path.read_text(encoding="utf-8"))
        previous_baseline_text = shadow_record["previous_baseline_text"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error(
            "[HallucinationRollback] Cannot roll back theory_baseline.md for %s - shadow memory "
            "malformed or missing 'previous_baseline_text': %s",
            cycle_dir.name,
            exc,
        )
        return False

    config.THEORY_BASELINE_FILE.write_text(previous_baseline_text, encoding="utf-8")
    logger.warning(
        "[HallucinationRollback] Rolled back theory_baseline.md to its pre-Theoretician state for %s.",
        cycle_dir.name,
    )
    return True


# --------------------------------------------------------------------------- #
# Cycle execution
# --------------------------------------------------------------------------- #

def run_cycle_sequence(cycle_num: int) -> CycleResult:
    """
    Run a single, complete research cycle end-to-end, in the full
    blueprint's strict order:

        a) Team 3   (Simulation)    : PythonArchitect -> SandboxDebugger
        b) Team 5   (Execution)     : CompilerMachinePlanner -> SemanticSafetyAgent
        c) [Hardware Mocking Hook]  : wait_for_hardware()
        d) Team 4   (Synthesis)     : SynthesisDataAnalyst -> ScientificHypothesisArchitect
                                       -> Gatekeeper (with in-cycle veto-retry, see
                                       _run_team_4_with_veto_retry)
        e) Team 2   (Theory)        : Theoretician -> FactChecker -> KnowledgeCurator
        f) ChefAgent (Coordinator)  : updates summary.md, decides goal_achieved

    Exceptions from any step are allowed to propagate OUT of this function
    (except GatekeeperVetoException retries, handled internally up to
    MAX_GATEKEEPER_RETRIES) - `main()`'s intelligent exception routing is
    what decides what to do about each kind of failure. This function's job
    is purely to run the pipeline in order; it does not itself decide how to
    recover from anything.

    Args:
        cycle_num: 1-indexed cycle number to run.

    Returns:
        A CycleResult bundling the CycleContext that was run and ChefAgent's
        goal_achieved verdict.
    """
    logger.info("=" * 70)
    logger.info("Starting research cycle #%d", cycle_num)
    logger.info("=" * 70)

    context = create_cycle_directory(cycle_num)
    _check_estop(f"start of cycle {cycle_num}")

    # --- a) Team 3: Simulation (Digital Twin) --------------------------------
    logger.info("--- Team 3: Simulation ---")
    PythonArchitect().execute(context.cycle_dir)
    SandboxDebugger().execute(context.cycle_dir)  # may raise SimulationFailedException
    _check_estop(f"after Team 3 for {context.cycle_dir.name}")

    # --- b) Team 5: Experiment Execution (The Compiler) ----------------------
    logger.info("--- Team 5: Execution ---")
    CompilerMachinePlanner().execute(context.cycle_dir)
    SemanticSafetyAgent().execute(context.cycle_dir)  # may raise HardwareSafetyException
    _check_estop(f"after Team 5 for {context.cycle_dir.name}")

    # --- c) Hardware Mocking Hook ----------------------------------------------
    logger.info("--- Hardware Mocking Hook ---")
    wait_for_hardware(context.cycle_dir)
    _check_estop(f"after hardware mock for {context.cycle_dir.name}")

    # --- d) Team 4: Synthesis & Hypothesis (with shortcut) ----------
    logger.info("--- Team 4: Synthesis ---")
    team4_exception = None
    try:
        _run_team_4_with_veto_retry(context.cycle_dir)
    except (GatekeeperVetoException, HardwareExecutionException) as exc:
        # Wir fangen auch die HardwareExecutionException ab!
        # Wir merken uns die Exception, werfen sie aber NOCH NICHT!
        team4_exception = exc
        logger.warning(
            "Team 4 raised a blocking exception (%s). Deferring to ChefAgent before routing.",
            type(exc).__name__
        )

    _check_estop(f"after Team 4 for {context.cycle_dir.name}")

    # --- e) Team 2: Theory (Nur wenn Team 4 keinen Hard Deadlock hatte) ---
    if team4_exception is None:
        logger.info("--- Team 2: Theory ---")
        Theoretician().execute(context.cycle_dir)
        FactChecker().execute(context.cycle_dir)
        KnowledgeCurator().execute(context.cycle_dir)
        _check_estop(f"after Team 2 for {context.cycle_dir.name}")
    else:
        logger.info("Skipping Team 2 (Theory) due to Team 4 blocking exception.")

    # --- f) ChefAgent: Coordinator ------------------------------------------------
    logger.info("--- ChefAgent: Coordination & Final Reporting ---")
    chef = ChefAgent()
    chef.execute(context.cycle_dir)

    if team4_exception is not None:
        raise team4_exception

    logger.info("Finished research cycle #%d.", cycle_num)
    logger.info("-" * 70)

    return CycleResult(context=context, goal_achieved=chef.goal_achieved)


# --------------------------------------------------------------------------- #
# Top-level bootstrap + entry point
# --------------------------------------------------------------------------- #

def bootstrap() -> WorkspaceManager:
    """Ensure the on-disk workspace exists and is correctly structured before any cycle runs."""
    logger.info("Self-Driving Lab starting up.")
    logger.info("Using Ollama model: %s", config.OLLAMA_MODEL)
    logger.info("Workspace root: %s", config.WORKSPACE_ROOT)

    workspace = WorkspaceManager()
    workspace.initialize()

    if not workspace.is_initialized():
        raise RuntimeError("Workspace failed to initialize correctly - aborting startup.")

    logger.info("Workspace verified and ready.")
    return workspace


def main(max_cycles: int = 1) -> None:
    """
    Process entry point. Runs research cycles starting from the next cycle
    number not yet present on disk, stopping when EITHER ChefAgent reports
    `goal_achieved=True`, OR `max_cycles` cycles have been run, whichever
    comes first.

    Intelligent Exception Routing:
    -----------------------------------
    `run_cycle_sequence()` is wrapped in a try/except with SEVERAL specific
    `except` clauses, each handling a genuinely different KIND of failure
    the way it actually warrants, before a final generic catch-all:

        0. `EmergencyStopException` (Step 8) - checked FIRST and handled
           completely differently from every other case below: this means
           an operator explicitly pressed the dashboard's Global E-Stop
           button. It is NOT a crisis to resolve - `DeadlockManager` is
           never invoked, and the loop does not proceed to the next cycle.
           `main()` logs the halt and returns immediately, leaving the
           E-Stop flag in place (it latches, like a physical E-Stop, until
           an operator explicitly clears it via the dashboard) so a
           subsequent `main_loop.py` invocation won't silently resume.
        1. `GatekeeperVetoException` - only ever reaches here after
           `_run_team_4_with_veto_retry` has ALREADY tried looping back to
           Team 3 up to MAX_GATEKEEPER_RETRIES times internally. By the time
           it's caught here, it represents a genuine, persistent disagreement
           worth a full DeadlockManager vote.
        2. `HallucinationException` - triggers a deterministic rollback of
           theory_baseline.md (see _rollback_theory_baseline), then is STILL
           routed to DeadlockManager (per design: the rollback fixes the
           FILE, but the lab still needs to decide what to do about the
           cycle as a whole).
        3. `SimulationFailedException` / `HardwareSafetyException` /
           `HardwareExecutionException` - each routed to DeadlockManager
           with an exception-specific reason string carrying the concrete
           violations/traceback, not just a generic message.
        4. Any other `Exception` (e.g. `LLMValidationException`, or an
           unexpected bug) - falls through to the same DeadlockManager
           routing as a last resort, with `str(exc)` as the reason.

    In every branch EXCEPT the E-Stop case, after DeadlockManager resolves
    the crisis, the loop proceeds to the NEXT cycle number (rather than
    retrying the same cycle_num) - see the original Step 8 design note on
    why: retrying the exact same cycle in a tight loop risks spinning
    forever on a persistent failure, and the failed cycle's directory
    already exists on disk, so `find_next_cycle_number()` naturally skips
    past it either way.
    """
    bootstrap()

    consecutive_success_vetoes = 0

    for _ in range(max_cycles):
        next_cycle_num = find_next_cycle_number()
        # Reconstructed independently of run_cycle_sequence()'s return value
        # so we know exactly where to route a failure even if the exception
        # occurs before that function gets a chance to return anything - the
        # directory itself is guaranteed to already exist by this point,
        # since create_cycle_directory() runs as the very first step inside
        # run_cycle_sequence(), before any agent that could fail.
        cycle_dir_for_this_attempt = config.RESEARCH_CYCLES_DIR / CYCLE_DIR_NAME_TEMPLATE.format(
            cycle_num=next_cycle_num
        )

        try:
            _check_estop(f"before starting cycle {next_cycle_num}")
            result = run_cycle_sequence(cycle_num=next_cycle_num)

        except EmergencyStopException as exc:
            logger.critical(
                "GLOBAL E-STOP ACTIVE (%s) - halting main_loop.py immediately at cycle #%d. "
                "Clear the E-Stop flag via the dashboard's Tab 1 before restarting.",
                exc.detail,
                next_cycle_num,
            )
            return

        except GatekeeperVetoException as exc:
            if exc.is_success_signal:
                consecutive_success_vetoes += 1
                logger.info(
                    "Cycle #%d: Gatekeeper VETO is a SUCCESS SIGNAL (%d consecutive).",
                    next_cycle_num, consecutive_success_vetoes,
                )
                if consecutive_success_vetoes >= 2:
                    logger.info("Gatekeeper has vetoed %d consecutive cycles citing goal achievement. Halting.", consecutive_success_vetoes)
                    break
                continue
            else:
                consecutive_success_vetoes = 0

            logger.error("Cycle #%d: Gatekeeper VETO persisted (Hard Deadlock).", next_cycle_num)
            reason = f"Gatekeeper vetoed a new physical experiment after {MAX_GATEKEEPER_RETRIES} attempts. Final reasoning: {exc.reasoning}"
            resolution: DeadlockResolution = DeadlockManager().resolve_deadlock(reason=reason, cycle_dir=cycle_dir_for_this_attempt)
            if resolution.action == "terminate_project": break
            continue

        except HallucinationException as exc:
            logger.error("Cycle #%d: FactChecker rejected theory_baseline.md: %s", next_cycle_num, exc.reasoning)
            rollback_succeeded = _rollback_theory_baseline(cycle_dir_for_this_attempt)
            reason = (
                f"FactChecker rejected the updated theory_baseline.md as scientifically invalid: "
                f"{exc.reasoning} (the baseline was "
                f"{'successfully' if rollback_succeeded else 'NOT'} rolled back to its exact "
                f"pre-Theoretician state)."
            )
            resolution = DeadlockManager().resolve_deadlock(reason=reason, cycle_dir=cycle_dir_for_this_attempt)
            logger.warning(
                "Deadlock resolved for cycle #%d -> %s (action=%s). Proceeding to the next cycle.",
                next_cycle_num, resolution, resolution.action,
            )
            continue

        except SimulationFailedException as exc:
            logger.error("Cycle #%d: simulation failed after %d attempts.", next_cycle_num, exc.attempts)
            stderr_tail = exc.last_stderr[-500:] if exc.last_stderr else "(empty)"
            reason = (
                f"The digital-twin simulation script failed to execute successfully after "
                f"{exc.attempts} execute-and-fix attempts. Last captured stderr: {stderr_tail}"
            )
            resolution = DeadlockManager().resolve_deadlock(reason=reason, cycle_dir=cycle_dir_for_this_attempt)
            logger.warning(
                "Deadlock resolved for cycle #%d -> %s (action=%s). Proceeding to the next cycle.",
                next_cycle_num, resolution, resolution.action,
            )
            continue

        except HardwareSafetyException as exc:
            logger.error("Cycle #%d: experiment job failed hardware safety checks.", next_cycle_num)
            reason = (
                f"The compiled experiment job failed {len(exc.violations)} deterministic hardware "
                f"safety check(s) and was not deployed: {'; '.join(exc.violations)}"
            )
            resolution = DeadlockManager().resolve_deadlock(reason=reason, cycle_dir=cycle_dir_for_this_attempt)
            logger.warning(
                "Deadlock resolved for cycle #%d -> %s (action=%s). Proceeding to the next cycle.",
                next_cycle_num, resolution, resolution.action,
            )
            continue

        except HardwareExecutionException as exc:
            logger.error("Cycle #%d: hardware reality check failed.", next_cycle_num)
            reason = (
                f"The deterministic hardware reality check failed with {len(exc.violations)} "
                f"violation(s): {'; '.join(exc.violations)}"
            )
            resolution = DeadlockManager().resolve_deadlock(reason=reason, cycle_dir=cycle_dir_for_this_attempt)
            logger.warning(
                "Deadlock resolved for cycle #%d -> %s (action=%s). Proceeding to the next cycle.",
                next_cycle_num, resolution, resolution.action,
            )
            continue

        except HardwareTimeoutException as exc:
            logger.error("Cycle #%d: hardware simulator timeout.", next_cycle_num)
            reason = str(exc)
            resolution = DeadlockManager().resolve_deadlock(reason=reason, cycle_dir=cycle_dir_for_this_attempt)
            logger.warning(
                "Deadlock resolved for cycle #%d -> %s (action=%s). Proceeding to the next cycle.",
                next_cycle_num, resolution, resolution.action,
            )
            continue

        except Exception as exc:
            # Generic last resort: LLMValidationException (an agent's LLM
            # output never passed schema validation after all retries), an
            # orchestration precondition bug surfacing as FileNotFoundError,
            # or anything else not given a more specific handler above.
            logger.error(
                "Cycle #%d failed with an unhandled exception - invoking DeadlockManager. Error: %s",
                next_cycle_num,
                exc,
                exc_info=True,
            )
            resolution = DeadlockManager().resolve_deadlock(reason=str(exc), cycle_dir=cycle_dir_for_this_attempt)
            logger.warning(
                "Deadlock resolved for cycle #%d -> %s (action=%s). Proceeding to the next cycle.",
                next_cycle_num, resolution, resolution.action,
            )
            continue

        if result.goal_achieved:
            logger.info("Directive satisfied at %s - halting further cycles.", result.context.cycle_dir.name)
            break
    else:
        logger.info(
            "Reached max_cycles=%d without the directive being satisfied. "
            "Run main_loop.py again to continue from the next cycle.",
            max_cycles,
        )

    logger.info("Step 5 (Grand Orchestration) complete.")


if __name__ == "__main__":
    main(max_cycles=5)
