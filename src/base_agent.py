"""
base_agent.py
==============
Abstract base class for every agent in the Self-Driving Lab.

Design philosophy - why agents are STATELESS and SYNCHRONOUS:
----------------------------------------------------------------
This system's "memory" lives entirely on disk (the workspace). Agents are not
long-lived objects with accumulated in-memory state; they are instantiated
fresh, run exactly once against a specific cycle directory, and then discarded.
This has two major benefits for a self-driving lab that may run unattended for
long periods:

  1. Crash isolation: if one agent throws an exception mid-cycle, the process
     can be restarted and simply re-read whatever the previous agent already
     wrote to disk - there's no in-memory state to lose or reconcile.
  2. Debuggability: every agent's entire "world view" for a given run is
     fully reconstructable by reading the cycle directory's contents. Nothing
     is hidden in an object's attributes that outlive the run.

Agents are executed SEQUENTIALLY in `main_loop.py`, one after another, not
concurrently. Because of that, `run()` is defined as a plain synchronous
method rather than `async def`: there is no I/O-concurrency benefit to gain
from async here (we are never awaiting multiple agents at once), and keeping
the codebase synchronous avoids the complexity of mixing sync Ollama calls
with an event loop for no real payoff. If a future agent needs to fan out
many concurrent LLM calls *within* its own `run()`, it's free to spin up its
own asyncio.run(...) or thread pool internally - that's an implementation
detail local to that agent, not part of the public BaseAgent contract.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

from src import config
from src.llm_wrapper import ask_llm_with_validation

# Generic type var so `self.ask_llm(..., response_model=SomeModel)` gives
# callers back a properly-typed `SomeModel` instance, mirroring the typing
# ergonomics already established in llm_wrapper.py.
T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Step 8: UI-Backend integration helpers
# --------------------------------------------------------------------------- #
# These are module-level (not BaseAgent methods) because they're pure,
# self-contained functions with no dependency on a particular agent
# instance - keeping them outside the class makes it obvious they don't
# participate in subclassing/inheritance at all.

def _load_ui_settings() -> dict:
    """
    Best-effort read of `config.UI_SETTINGS_FILE` (00_System/ui_settings.json),
    written by the dashboard's Tab 4 (see src/ui_dashboard.py). Returns {} if
    the file doesn't exist, is empty, or fails to parse.

    This is an OPTIONAL override layer, never a hard dependency: every agent
    must keep working exactly as it did before Step 8 if the dashboard has
    never been run, so any failure to read/parse this file is treated
    identically to "no overrides configured" rather than an error.
    """
    try:
        if not config.UI_SETTINGS_FILE.exists():
            return {}
        text = config.UI_SETTINGS_FILE.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read UI settings from %s: %s", config.UI_SETTINGS_FILE, exc)
        return {}


def _resolve_model_override(agent_name: str, requested_model: str) -> str:
    """
    Resolve the effective model for an agent, consulting the dashboard's
    per-team model assignment (Tab 4) as an override layer on top of
    `requested_model`.

    Resolution rule: if `requested_model` is anything OTHER than the plain
    system default (`config.OLLAMA_MODEL`), it is treated as an intentional,
    explicit choice and returned UNCHANGED - the UI override never
    silently supersedes a caller that asked for something specific (e.g.
    `DeadlockManager` constructing a voting agent with its own already-
    resolved `self.model`). Only when `requested_model` IS the plain
    default - which is what every agent's own `__init__` signature falls
    back to when its caller specifies nothing at all, i.e. the overwhelming
    majority of real construction calls throughout this codebase - do we
    consult `ui_settings.json` for a more specific, per-agent-name choice.

    This achieves "Tab 4 settings actually drive ask_llm()" without needing
    to touch every single team module's constructor signature (which would
    have meant editing ~11 files across the whole blueprint for one small
    integration): every existing `SomeAgent()` call, unchanged, now
    automatically picks up its team's configured model at construction
    time, re-read fresh every cycle since agents are constructed fresh
    every cycle (see module docstring's statelessness rationale).
    """
    if requested_model != config.OLLAMA_MODEL:
        return requested_model

    settings = _load_ui_settings()
    model_assignments = settings.get("model_assignments", {})
    override = model_assignments.get(agent_name) if isinstance(model_assignments, dict) else None
    return override if isinstance(override, str) and override.strip() else requested_model


def _write_active_agent_status(agent_name: str | None, cycle_dir: Path, status: str) -> None:
    """
    Write `config.ACTIVE_AGENT_STATUS_FILE` (00_System/active_agent.json) -
    a real-time heartbeat of which agent is currently executing and against
    which cycle. This is Step 8's "True Active Agent Telemetry": a
    definitive, live signal that `src/ui_dashboard.py`'s Mission Control tab
    reads directly, replacing the prior best-effort inference from which
    artifacts already exist on disk.

    Called from `BaseAgent.execute()` at the very start (status="running")
    and unconditionally in its `finally` block (status="idle") - so the file
    always reflects reality even if `run()` raises, and there is never a
    stale "running" entry left behind after a crash.

    Failure to write this file must NEVER take down the actual agent work it
    describes - telemetry is a nice-to-have overlay, not a dependency of the
    pipeline's correctness - so any OSError here is logged at debug level
    and swallowed rather than propagated.
    """
    try:
        config.SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent_name": agent_name,
            "status": status,
            "cycle_id": cycle_dir.name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        config.ACTIVE_AGENT_STATUS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write active-agent telemetry to %s: %s", config.ACTIVE_AGENT_STATUS_FILE, exc)


class BaseAgent(ABC):
    """
    Abstract base class every concrete agent (Planner, Researcher, Hardware
    Executor, Analyst, Chef, ...) must subclass.

    Contract for subclasses:
        - Implement `run(self, cycle_dir: Path) -> None`.
        - `run()` should be a single, self-contained pass: read whatever it
          needs from `cycle_dir` / the workspace, do its work (optionally via
          `self.ask_llm(...)`), and write its output back to disk. It should
          not assume it will be called again, and should not retain state
          across separate agent instantiations.
        - `run()` should raise on unrecoverable errors rather than silently
          swallowing them - `main_loop.py` is responsible for deciding how to
          handle a failed agent (retry the cycle, halt, alert, etc.), not the
          agent itself.

    Usage:
        class PlannerAgent(BaseAgent):
            def run(self, cycle_dir: Path) -> None:
                plan = self.ask_llm("Plan the next experiment...", response_model=PlanSchema)
                (cycle_dir / "A_Simulation" / "plan.json").write_text(plan.model_dump_json())

        agent = PlannerAgent(agent_name="Planner", workspace_path=config.WORKSPACE_ROOT)
        agent.run(cycle_dir=some_cycle_dir)
    """

    def __init__(
        self,
        agent_name: str,
        workspace_path: Path | None = None,
        model: str = config.OLLAMA_MODEL,
    ) -> None:
        """
        Args:
            agent_name: Human-readable identifier for this agent, used purely
                        for logging/traceability (e.g. "Planner", "Chef").
                        Not used for any control-flow decisions.
            workspace_path: Root of the shared workspace this agent operates
                             against. Defaults to config.WORKSPACE_ROOT so
                             most agents don't need to pass anything, while
                             still allowing dependency injection (e.g. for
                             tests that spin up an isolated workspace).
            model: Which Ollama model tag this agent's LLM calls should use.
                   Defaults to the system-wide default but can be overridden
                   per-agent (e.g. a lightweight agent using a smaller model).
                   Step 8: if left at the plain system default, this is
                   further resolved against the dashboard's per-team model
                   assignment (Tab 4) if one is configured - see
                   `_resolve_model_override`'s docstring for the exact rule.
        """
        if not agent_name or not agent_name.strip():
            # Fail loudly at construction time rather than producing confusing,
            # unattributable log lines later ("[] starting up...").
            raise ValueError("agent_name must be a non-empty string.")

        self.agent_name: str = agent_name.strip()
        self.workspace_path: Path = (workspace_path or config.WORKSPACE_ROOT).resolve()
        self.model: str = _resolve_model_override(self.agent_name, model)

        # Give every agent its own named logger (e.g. "agent.Planner") so log
        # output is easy to filter/grep per-agent, while still inheriting the
        # root logging configuration set up in main_loop.py.
        self.logger = logging.getLogger(f"agent.{self.agent_name}")

    # ------------------------------------------------------------------ #
    # Abstract contract
    # ------------------------------------------------------------------ #

    @abstractmethod
    def run(self, cycle_dir: Path) -> None:
        """
        Execute this agent's single pass of work for the given research cycle.

        Args:
            cycle_dir: Absolute path to the current cycle's directory, e.g.
                       /workspace/02_Research_Cycles/Cycle_001/. Subclasses
                       are expected to read/write within this directory's
                       subfolders (A_Simulation, B_Hardware, C_Analysis,
                       D_Shadow_Memory) as appropriate to their role.

        Must be implemented by every concrete subclass. Implementations
        should raise on unrecoverable failure rather than returning silently,
        so the orchestrator in main_loop.py can react appropriately.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Public helpers available to all subclasses
    # ------------------------------------------------------------------ #

    def execute(self, cycle_dir: Path) -> None:
        """
        Wrapper around `run()` that adds standardized entry/exit logging,
        error handling, and (Step 8) real-time active-agent telemetry. This
        is the method `main_loop.py` should actually call, rather than
        calling `run()` directly - it guarantees every agent's execution is
        bracketed by consistent, greppable log lines regardless of what the
        subclass's `run()` does internally.
        """
        self.logger.info("Entering agent '%s' | cycle_dir=%s", self.agent_name, cycle_dir)
        _write_active_agent_status(self.agent_name, cycle_dir, status="running")

        try:
            self.run(cycle_dir)
        except Exception:
            # Log with full traceback before re-raising, so main_loop.py can
            # decide how to handle the failure (e.g. halt the cycle) without
            # needing to duplicate error-logging logic in every agent.
            self.logger.exception("Agent '%s' raised an unhandled exception.", self.agent_name)
            raise
        else:
            self.logger.info("Exiting agent '%s' cleanly.", self.agent_name)
        finally:
            # Unconditional: this must run whether run() succeeded, raised,
            # or even if a signal interrupted it - the telemetry file should
            # never be left claiming an agent is "running" after it no
            # longer is, regardless of how execute() actually exited.
            _write_active_agent_status(None, cycle_dir, status="idle")

    def ask_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        response_model: Type[T] | None = None,
        max_retries: int = config.DEFAULT_MAX_RETRIES,
        temperature: float = config.DEFAULT_TEMPERATURE,
        context_size: int | None = None,
    ) -> str | T:
        """
        Convenience wrapper around `ask_llm_with_validation()` that:
          - Automatically uses this agent's configured `self.model`.
          - Adds standardized logging around prompt size and outcome, so
            token/prompt handling is visible in logs without every agent
            re-implementing it.
          - (Step 8) Resolves `temperature` and `context_size` dynamically
            against the dashboard's Tab 4 settings on EVERY call (not just
            once at construction time, unlike `self.model`) - since
            `ui_settings.json` can change between two calls within the same
            agent's `run()`, or between two agents in the same cycle,
            re-reading it fresh per call is what makes the dashboard's
            sliders feel "live" rather than only applying to agents
            constructed after a settings change.

        Resolution rule for `temperature`: mirrors `_resolve_model_override`'s
        logic exactly - if the caller passed something other than the plain
        system default (`config.DEFAULT_TEMPERATURE`), that explicit value is
        used as-is; otherwise `ui_settings.json`'s `temperature` is consulted.
        `context_size` has no pre-existing default to compare against, so it
        simply uses the caller's value if given, else `ui_settings.json`'s
        `context_size`, else `config.DEFAULT_CONTEXT_SIZE`.

        Args and return type otherwise mirror `ask_llm_with_validation()`
        exactly - see llm_wrapper.py for full behavior (auto-correction
        retries, markdown-fence stripping, LLMValidationException on
        exhaustion).
        """
        ui_settings = _load_ui_settings()

        effective_temperature = temperature
        if temperature == config.DEFAULT_TEMPERATURE:
            override_temperature = ui_settings.get("temperature")
            if isinstance(override_temperature, (int, float)):
                effective_temperature = float(override_temperature)

        effective_context_size = context_size
        if effective_context_size is None:
            override_context_size = ui_settings.get("context_size")
            effective_context_size = (
                int(override_context_size)
                if isinstance(override_context_size, (int, float))
                else config.DEFAULT_CONTEXT_SIZE
            )

        # Rough, dependency-free size signal for logs. This is NOT a real
        # tokenizer count (that would require pulling in a model-specific
        # tokenizer) - it's a cheap proxy so we can spot suspiciously huge
        # prompts in logs without adding a heavy dependency at this stage.
        approx_prompt_chars = len(prompt) + len(system_prompt)
        self.logger.debug(
            "[%s] Dispatching LLM call | model=%s | temperature=%.2f | context_size=%d | "
            "response_model=%s | ~%d prompt chars",
            self.agent_name,
            self.model,
            effective_temperature,
            effective_context_size,
            response_model.__name__ if response_model else "None (raw text)",
            approx_prompt_chars,
        )

        result = ask_llm_with_validation(
            prompt=prompt,
            system_prompt=system_prompt,
            response_model=response_model,
            max_retries=max_retries,
            model=self.model,
            temperature=effective_temperature,
            context_size=effective_context_size,
        )

        if isinstance(result, BaseModel):
            self.logger.debug(
                "[%s] LLM call succeeded | validated as %s", self.agent_name, type(result).__name__
            )
        else:
            self.logger.debug(
                "[%s] LLM call succeeded | ~%d response chars", self.agent_name, len(result)
            )

        return result

    def __repr__(self) -> str:
        # Helpful when agents show up in tracebacks / debugger output.
        return f"<{type(self).__name__} agent_name={self.agent_name!r} model={self.model!r}>"
