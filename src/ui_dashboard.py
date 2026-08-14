"""
ui_dashboard.py
=================
Step 7 - The Passive Observer: a NiceGUI web dashboard for the Self-Driving Lab.

Role in the system:
---------------------
This module is a completely SEPARATE process from `main_loop.py`. It does
NOT import `BaseAgent`, does NOT call `ask_llm`, and never invokes any
agent's `run()`/`execute()` method - it is a pure, passive observer of the
same on-disk workspace `main_loop.py` writes to. It reads whatever files
happen to exist, on a periodic timer, and renders them. This is exactly the
same "filesystem as shared memory" philosophy the rest of this project is
built on (see `src/base_agent.py`'s module docstring) - the dashboard is
just one more reader of that shared memory, alongside every agent.

Because it's a separate process, there is NO shared Python state between
this UI and the running (or not-yet-running) `main_loop.py` process. Every
interaction this dashboard offers is therefore implemented as either:
  (a) a pure file READ (almost everything - the 4 tabs' displays), or
  (b) a pure file WRITE that `main_loop.py` or an agent would need to
      independently notice on its own next pass (the Co-Pilot chat input,
      which appends to `copilot_feedback.md` for Gatekeeper to read next
      cycle; the Global E-Stop button, which writes a flag file - see the
      "Known limitations" section below for exactly what does and doesn't
      consume that flag today).

Launching the dashboard:
----------------------------
This is a standalone process, run separately from (and typically alongside)
`main_loop.py`:

    python src/ui_dashboard.py

It starts a local web server (NiceGUI, built on FastAPI/uvicorn) and opens a
browser tab automatically. By default it listens on http://localhost:8080 -
override with the `UI_DASHBOARD_PORT` environment variable if that port is
taken. You can run this in one terminal and `python main_loop.py` in
another; they only ever communicate through the workspace filesystem, so
the order you start them in doesn't matter, and either can be restarted
independently without disturbing the other.

Known limitations (read before relying on this for anything safety-critical):
-----------------------------------------------------------------------------
As of Step 8, three of the four gaps originally documented here have been
closed with real backend hooks:

- **"Active agent" is now LIVE, not inferred** - `BaseAgent.execute()`
  writes `config.ACTIVE_AGENT_STATUS_FILE` (00_System/active_agent.json) at
  the start of every agent invocation and unconditionally clears it to
  "idle" in a `finally` block, so it can never be left stuck claiming an
  agent is running after it no longer is. `_read_active_agent_status()`
  reads this directly. The older artifact-based inference
  (`_stage_completion` / `_current_active_stage_name`) is KEPT and still
  used for the done/pending coloring of each Kanban card, and as a graceful
  fallback for the brief window before any agent has ever run, or against a
  workspace produced by a pre-Step-8 codebase that never wrote this file.
- **The Global E-Stop button now genuinely halts `main_loop.py`** -
  `run_cycle_sequence()` checks `config.ESTOP_FLAG_FILE` at the start of
  every cycle and between every team boundary, raising
  `EmergencyStopException` the moment it's found; `main()` catches that
  specifically and halts immediately (see `main_loop.py`'s module and
  `main()` docstrings). The flag still latches (like a physical E-Stop)
  until explicitly cleared via this dashboard's "Clear E-Stop" button.
- **Tab 4's settings now genuinely drive `ask_llm()`** - `BaseAgent.__init__`
  resolves per-team model assignment from `config.UI_SETTINGS_FILE`
  whenever a caller left `model` at the plain system default (which is what
  every agent's own constructor falls back to when unspecified - the
  overwhelming majority of real construction calls throughout this
  codebase); `BaseAgent.ask_llm` resolves `temperature`/`context_size` the
  same way, freshly on EVERY call, so a slider change is picked up by the
  very next LLM call, not just newly-constructed agents. An explicit,
  non-default value passed by a caller is still always respected as-is -
  see `_resolve_model_override`'s docstring in `src/base_agent.py` for the
  exact rule and its rationale.

One limitation remains, unchanged from before:

4. **The 4-station carousel pulse is a stage-level animation, not literal
   per-station telemetry.** Nothing in the pipeline currently records which
   of the 4 physical stations is active at a given instant - the mocked
   `wait_for_hardware()` hook completes all 4 stations' worth of work
   instantly. The carousel therefore pulses ALL 4 station cards together, in
   a staggered wave, while a job is queued and not yet reflected in that
   cycle's `hardware_protocol.json`/`measurement.csv` - representing "the
   carousel is busy" rather than claiming station-level precision it
   doesn't have.

Every read-only element (Mission Control's feed and stage indicator, the
hardware queue table, the Lab Journal's plots and reports, the settings
persistence) reflects real, current on-disk state, and as of Step 8 every
WRITE-side hook (E-Stop, per-team model/temperature/context-size, Co-Pilot
feedback) has a real backend consumer too - remaining limitation #4 is
purely a "we don't have this data at all" gap, not an unwired connection.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import ui

from src import config

logger = logging.getLogger("ui_dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

POLL_INTERVAL_SECONDS = 2.0

# Note (Step 8): the E-Stop flag, UI settings, and active-agent telemetry
# paths are now defined ONCE in src/config.py (ESTOP_FLAG_FILE,
# UI_SETTINGS_FILE, ACTIVE_AGENT_STATUS_FILE) and referenced directly from
# there throughout this module, rather than each side (this dashboard vs.
# main_loop.py/base_agent.py) independently reconstructing the same path -
# see config.py's "UI-BACKEND INTEGRATION FILES" section for the rationale.

COPILOT_FEEDBACK_FILENAME = "copilot_feedback.md"

CYCLE_DIR_NAME_PATTERN = re.compile(r"^Cycle_(\d+)$")

# Agent names Tab 4 offers per-team model overrides for - matches the
# `agent_name` strings each BaseAgent subclass registers itself under, so
# these labels line up with what actually appears in shadow memory/logs.
ASSIGNABLE_AGENT_NAMES: tuple[str, ...] = (
    "PythonArchitect",
    "MachinePlanner",
    "SynthesisDataAnalyst",  # team_4_synthesis.DataAnalyst, as aliased in main_loop.py
    "HypothesisArchitect",
    "Gatekeeper",
    "Theoretician",
    "FactChecker",
    "KnowledgeCurator",
    "ChefAgent",
)

# The visual labels for the 4-station carousel (Tab 2).
STATION_LABELS: tuple[str, ...] = (
    "Station 1\nReagents",
    "Station 2\nProcess",
    "Station 3\nAnalytics",
    "Station 4\nCleanup",
)


# --------------------------------------------------------------------------- #
# Safe filesystem helpers
# --------------------------------------------------------------------------- #
# Every read in this module goes through one of these, because this process
# has no coordination with whatever agent might be mid-write to the exact
# file being polled at the exact moment we poll it (agents use plain
# `Path.write_text()`, which is not atomic). Rather than trying to detect or
# prevent that race, every helper here just tolerates it: on any read error,
# return a clearly-labeled placeholder and let the NEXT poll (2 seconds
# later) pick up the settled content. A dashboard that occasionally shows a
# one-tick-stale placeholder is a completely acceptable trade-off for a
# passive observer; a dashboard that crashes because it caught a file
# half-written is not.

def _safe_read_text(path: Path, default: str = "") -> str:
    try:
        if not path.exists():
            return default
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("Transient read error for %s (will retry next poll): %s", path, exc)
        return default


def _safe_read_json(path: Path) -> dict | list | None:
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Transient/parse read error for %s (will retry next poll): %s", path, exc)
        return None


def _safe_iterdir(path: Path) -> list[Path]:
    try:
        if not path.exists():
            return []
        return sorted(path.iterdir())
    except OSError as exc:
        logger.debug("Transient listdir error for %s (will retry next poll): %s", path, exc)
        return []


# --------------------------------------------------------------------------- #
# Cycle discovery & stage inference
# --------------------------------------------------------------------------- #

def _find_latest_cycle_dir() -> Path | None:
    """
    Return the highest-numbered `Cycle_XXX` directory currently on disk, or
    None if no cycle has been created yet. Mirrors `main_loop.py`'s
    `find_next_cycle_number` scanning logic, but returns the LATEST
    *existing* cycle (what the dashboard should be showing right now)
    rather than the next number to create.
    """
    highest_num = 0
    highest_dir: Path | None = None
    for entry in _safe_iterdir(config.RESEARCH_CYCLES_DIR):
        if not entry.is_dir():
            continue
        match = CYCLE_DIR_NAME_PATTERN.match(entry.name)
        if not match:
            continue
        num = int(match.group(1))
        if num > highest_num:
            highest_num = num
            highest_dir = entry
    return highest_dir


def _stage_completion(cycle_dir: Path) -> list[tuple[str, bool]]:
    """
    Best-effort inference of which of the 6 pipeline stages have completed
    for this cycle, in strict execution order. See module docstring's
    "Known limitations" #1.
    """
    shadow_dir = cycle_dir / "D_Shadow_Memory"
    hardware_dir = cycle_dir / "B_Hardware"
    analysis_dir = cycle_dir / "C_Analysis"

    team5_shadow = _safe_read_json(shadow_dir / "team5_semantic_safety_shadow.json")
    team5_deployed = bool(isinstance(team5_shadow, dict) and team5_shadow.get("deployed"))

    return [
        ("Team 3: Simulation", (cycle_dir / "A_Simulation" / "sim_data.csv").exists()),
        ("Team 5: Execution", team5_deployed),
        (
            "Hardware",
            (hardware_dir / "measurement.csv").exists() and (hardware_dir / "hardware_protocol.json").exists(),
        ),
        ("Team 4: Synthesis", (analysis_dir / "gatekeeper_decision.json").exists()),
        ("Team 2: Theory", (shadow_dir / "team2_knowledge_curator_shadow.json").exists()),
        ("ChefAgent", (shadow_dir / "chef_shadow.json").exists()),
    ]


def _current_active_stage_name(stage_completion: list[tuple[str, bool]]) -> str:
    """The first not-yet-done stage, or 'Cycle Complete' if every stage is done."""
    for name, done in stage_completion:
        if not done:
            return name
    return "Cycle Complete"


# Maps the exact `agent_name` string each concrete agent registers itself
# under (see each class's `super().__init__(agent_name=..., ...)`) to the
# Kanban stage label it belongs to. Used by `_read_active_agent_status` to
# translate the definitive telemetry file's live agent_name into "which
# stage card should show as ACTIVE". Note `Chef`, not `ChefAgent` - the
# latter is only `main_loop.py`'s import ALIAS for `src.agents.chef.Chef`;
# the class itself registers the literal string "Chef" as its agent_name.
AGENT_NAME_TO_STAGE: dict[str, str] = {
    "PythonArchitect": "Team 3: Simulation",
    "SandboxDebugger": "Team 3: Simulation",
    "MachinePlanner": "Team 5: Execution",
    "SemanticSafetyAgent": "Team 5: Execution",
    "DataAnalyst": "Team 4: Synthesis",
    "HypothesisArchitect": "Team 4: Synthesis",
    "Gatekeeper": "Team 4: Synthesis",
    "Theoretician": "Team 2: Theory",
    "FactChecker": "Team 2: Theory",
    "KnowledgeCurator": "Team 2: Theory",
    "Chef": "ChefAgent",
}


def _read_active_agent_status() -> tuple[str | None, str]:
    """
    Read `config.ACTIVE_AGENT_STATUS_FILE` (00_System/active_agent.json) -
    the definitive, real-time heartbeat written by `BaseAgent.execute()`
    (Step 8) - and return `(agent_name, status)`. `agent_name` is None
    whenever `status != "running"` (i.e. idle, or the file is missing/
    unreadable/predates this telemetry existing), so callers can treat "no
    agent_name" as the single unambiguous signal for "nothing is definitively
    known to be running right now" without needing to separately check status.
    """
    payload = _safe_read_json(config.ACTIVE_AGENT_STATUS_FILE)
    if not isinstance(payload, dict):
        return None, "unknown"
    status = payload.get("status", "unknown")
    agent_name = payload.get("agent_name") if status == "running" else None
    return agent_name, status


def _carousel_state(latest_cycle_dir: Path | None) -> str:
    """
    Coarse carousel state for Tab 2's visualization: 'idle' (nothing
    queued), 'processing' (a job is queued and that cycle's hardware output
    doesn't exist yet), or 'complete' (queued job's cycle already has
    hardware output). See module docstring's "Known limitations" #4 for why
    this is stage-level, not per-station.
    """
    queue_path = config.HARDWARE_QUEUE_DIR / "experiment.json"
    payload = _safe_read_json(queue_path)
    if not isinstance(payload, dict):
        return "idle"

    cycle_id = payload.get("cycle_id")
    if not cycle_id:
        return "processing"

    hardware_dir = config.RESEARCH_CYCLES_DIR / cycle_id / "B_Hardware"
    if (hardware_dir / "measurement.csv").exists() and (hardware_dir / "hardware_protocol.json").exists():
        return "complete"
    return "processing"


# --------------------------------------------------------------------------- #
# Global E-Stop
# --------------------------------------------------------------------------- #

def _is_estop_active() -> tuple[bool, str]:
    if not config.ESTOP_FLAG_FILE.exists():
        return False, ""
    return True, _safe_read_text(config.ESTOP_FLAG_FILE, default="(E-Stop flag present but unreadable.)")


def _trigger_estop(reason: str) -> None:
    """
    Write the global E-Stop flag file (config.ESTOP_FLAG_FILE). As of Step 8,
    `main_loop.py`'s `run_cycle_sequence()` checks this file at the start of
    every cycle and between every team boundary (see its `_check_estop`
    helper and `EmergencyStopException`) - triggering it here now genuinely
    halts a running loop, not just records an unconsumed intent.
    """
    config.ESTOP_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    config.ESTOP_FLAG_FILE.write_text(
        f"E-STOP TRIGGERED at {timestamp}\nReason: {reason or '(no reason given)'}\n", encoding="utf-8"
    )
    logger.critical("[E-Stop] TRIGGERED. Reason: %s", reason)


def _clear_estop() -> None:
    if config.ESTOP_FLAG_FILE.exists():
        config.ESTOP_FLAG_FILE.unlink()
    logger.warning("[E-Stop] Cleared.")


# --------------------------------------------------------------------------- #
# UI settings persistence (Tab 4)
# --------------------------------------------------------------------------- #

def _default_ui_settings() -> dict[str, Any]:
    return {
        "model_assignments": {name: config.OLLAMA_MODEL for name in ASSIGNABLE_AGENT_NAMES},
        "temperature": config.DEFAULT_TEMPERATURE,
        "context_size": config.DEFAULT_CONTEXT_SIZE,
    }


def _load_ui_settings() -> dict[str, Any]:
    settings = _safe_read_json(config.UI_SETTINGS_FILE)
    defaults = _default_ui_settings()
    if not isinstance(settings, dict):
        return defaults
    # Merge over defaults so a settings file saved by an older version of
    # this dashboard (missing a newly-added key, or missing a newly-added
    # assignable agent) doesn't crash the UI - unknown/missing keys just
    # fall back to their default rather than raising a KeyError.
    merged = {**defaults, **settings}
    merged["model_assignments"] = {**defaults["model_assignments"], **settings.get("model_assignments", {})}
    return merged


def _save_ui_settings(settings: dict[str, Any]) -> None:
    config.UI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.UI_SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Ollama model discovery (Tab 4)
# --------------------------------------------------------------------------- #

def _list_available_ollama_models() -> list[str]:
    """
    Query the local Ollama server for installed model tags, for the Tab 4
    dropdowns. Wrapped defensively: if no Ollama server is reachable (e.g.
    this dashboard is being run/tested without one, exactly as throughout
    this project's own test suite), fall back to just `config.OLLAMA_MODEL`
    so the dropdown is never empty.
    """
    try:
        import ollama

        response = ollama.list()
        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])
        models = models or []

        names: list[str] = []
        for m in models:
            name = getattr(m, "model", None)
            if name is None and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if name:
                names.append(name)
        return names or [config.OLLAMA_MODEL]
    except Exception as exc:
        logger.info("Could not reach Ollama server to list models (using configured default only): %s", exc)
        return [config.OLLAMA_MODEL]


# --------------------------------------------------------------------------- #
# Hardware load indicators (Tab 4)
# --------------------------------------------------------------------------- #

def _read_hardware_load() -> tuple[float | None, float | None]:
    """Return (cpu_percent, ram_percent), or (None, None) if psutil is unavailable."""
    try:
        import psutil

        return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
    except Exception as exc:
        logger.debug("psutil unavailable for hardware load indicators: %s", exc)
        return None, None


# --------------------------------------------------------------------------- #
# CSV plotting helper (Tab 3)
# --------------------------------------------------------------------------- #

def _add_csv_to_subplot(fig: go.Figure, path: Path, row: int, col: int) -> bool:
    """Liest eine CSV und zeichnet ALLE numerischen Spalten als eigene Linien in den Subplot."""
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
        csv_rows = list(csv.reader(io.StringIO(text)))
    except OSError:
        return False

    if len(csv_rows) < 2 or len(csv_rows[0]) < 2:
        return False

    headers = csv_rows[0]
    data = csv_rows[1:]

    # Spalte 0 wird immer als X-Achse verwendet
    x_vals = [r[0] for r in data if len(r) > 0]
    has_data = False

    # Iteriere über alle restlichen Spalten und plotte sie auf der Y-Achse
    for i in range(1, len(headers)):
        y_vals = []
        for r in data:
            if len(r) > i:
                try:
                    y_vals.append(float(r[i]))
                except ValueError:
                    y_vals.append(None)
            else:
                y_vals.append(None)

        if any(v is not None for v in y_vals):
            fig.add_trace(
                go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name=f"{path.name[:3]}: {headers[i]}"),
                row=row, col=col
            )
            has_data = True

    return has_data


def _build_overlay_figure(cycle_dir: Path) -> go.Figure:
    """Baut einen Side-by-Side Plot, damit unterschiedliche X-Achsen sich nicht mehr zerquetschen."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Simulated Prediction", "Actual Measurement")
    )

    has_sim = _add_csv_to_subplot(fig, cycle_dir / "A_Simulation" / "sim_data.csv", row=1, col=1)
    has_actual = _add_csv_to_subplot(fig, cycle_dir / "B_Hardware" / "measurement.csv", row=1, col=2)

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        template="plotly_dark",  # Passt optisch perfekt zum Dark-Mode des Dashboards!
        legend=dict(orientation="h", y=-0.2)  # Legende rutscht nach unten
    )

    if not (has_sim or has_actual):
        fig.add_annotation(
            text="No plottable data yet for this cycle.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray")
        )

    return fig


# --------------------------------------------------------------------------- #
# Co-Pilot feedback (Tab 3)
# --------------------------------------------------------------------------- #

def _append_copilot_feedback(cycle_dir: Path, message: str) -> None:
    """
    Append a timestamped Co-Pilot note to this cycle's
    `C_Analysis/copilot_feedback.md`, for Gatekeeper (Team 4.3) to read on
    its next invocation - see `Gatekeeper._read_copilot_feedback_if_present`
    in `src/team_4_synthesis.py`. Appended (not overwritten) so multiple
    notes submitted across the same cycle all accumulate, matching the
    "growing log" pattern this codebase uses elsewhere (e.g. FactChecker's
    fact_check_log.json).
    """
    analysis_dir = cycle_dir / "C_Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = analysis_dir / COPILOT_FEEDBACK_FILENAME

    timestamp = datetime.now(timezone.utc).isoformat()
    entry = f"\n## Co-Pilot Note ({timestamp})\n{message.strip()}\n"
    with feedback_path.open("a", encoding="utf-8") as f:
        f.write(entry)

    logger.info("[Co-Pilot] Appended note to %s", feedback_path)


# --------------------------------------------------------------------------- #
# UI construction
# --------------------------------------------------------------------------- #
# NiceGUI idiom followed throughout: build every element ONCE up front and
# keep a reference to it; the single `_refresh_all()` timer callback below
# then MUTATES those existing elements' `.text`/`.content`/`.value`
# properties on every poll, rather than tearing down and rebuilding the page
# every tick. This avoids flicker and keeps the browser-side DOM stable
# (e.g. scroll position in the summary feed is preserved across refreshes).

# Note: `refs` is NOT a module-level global (see `index()` below) - each
# client connection gets its own fresh dict, threaded explicitly as a
# parameter through every build/refresh function. See the correctness note
# in `index()`'s docstring for why this matters.


def _build_tab_1_mission_control(refs: dict[str, Any]) -> None:
    """Tab 1: Mission Control - Kanban-style stage indicator, summary feed, Global E-Stop."""
    with ui.column().classes("w-full gap-4"):
        ui.label("Mission Control").classes("text-2xl font-bold")

        # --- Definitive active-agent readout (Step 8) -------------------------------
        # Reads config.ACTIVE_AGENT_STATUS_FILE directly - a real-time heartbeat
        # written by BaseAgent.execute() itself, not inferred from artifacts.
        refs["active_agent_label"] = ui.label("Currently Running: (checking...)").classes(
            "text-base font-semibold"
        )

        # --- Kanban-style pipeline stage strip -------------------------------------
        ui.label("Pipeline Stage").classes("text-lg font-semibold")
        with ui.row().classes("w-full gap-2 flex-wrap") as stage_row:
            refs["stage_row"] = stage_row
            refs["stage_cards"] = {}  # populated lazily in _refresh_tab_1 as stage names appear

        ui.separator()

        # --- Global E-Stop -----------------------------------------------------------
        with ui.card().classes("w-full bg-red-50 border-2 border-red-300"):
            ui.label("⛔ Global Software E-Stop").classes("text-lg font-bold text-red-700")
            refs["estop_status_label"] = ui.label("Status: checking...").classes("text-sm")

            def do_trigger() -> None:
                _trigger_estop(reason=refs["estop_reason_input"].value or "")
                ui.notify(
                    "E-Stop flag written. main_loop.py will halt at the next cycle boundary or "
                    "team-execution boundary it checks (within one pipeline step).",
                    type="warning",
                    timeout=6000,
                )

            def do_clear() -> None:
                _clear_estop()
                ui.notify("E-Stop flag cleared.", type="positive")

            refs["estop_reason_input"] = ui.input(label="Reason (optional)").classes("w-full")
            with ui.row().classes("gap-2 mt-2"):
                ui.button("TRIGGER E-STOP", color="red", on_click=do_trigger).props("icon=warning")
                ui.button("Clear E-Stop", color="grey", on_click=do_clear).props("outline")

        ui.separator()

        # --- Chef's summary.md scrolling feed -----------------------------------------
        ui.label("Global Summary Log (00_System/summary.md)").classes("text-lg font-semibold")
        with ui.scroll_area().classes("w-full h-80 border rounded p-2"):
            refs["summary_markdown"] = ui.markdown("*(loading...)*")


def _build_tab_2_hardware_monitor(refs: dict[str, Any]) -> None:
    """Tab 2: Hardware Live-Monitor - 4-station carousel visual, hardware queue table."""
    ui.add_head_html(
        """
    <style>
      @keyframes station-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.55); }
        50% { box-shadow: 0 0 0 12px rgba(59,130,246,0); }
      }
      .station-pulsing { animation: station-pulse 1.4s ease-in-out infinite; }
    </style>
    """
    )

    with ui.column().classes("w-full gap-4"):
        ui.label("Hardware Live-Monitor").classes("text-2xl font-bold")

        ui.label("4-Station Carousel").classes("text-lg font-semibold")
        ui.label(
            "Pulses represent the carousel as a whole being busy (a job is queued but not yet "
            "reflected in hardware output) - not literal per-station telemetry. See module "
            "docstring 'Known limitations' #4."
        ).classes("text-xs text-gray-500")
        refs["carousel_state_label"] = ui.label("State: idle")
        with ui.row().classes("w-full gap-4 justify-center"):
            refs["station_cards"] = []
            for label in STATION_LABELS:
                with ui.card().classes("w-40 h-28 items-center justify-center") as card:
                    ui.label(label).classes("text-center whitespace-pre-line font-medium")
                refs["station_cards"].append(card)

        ui.separator()

        ui.label("Hardware Queue (03_Hardware_Queue/)").classes("text-lg font-semibold")
        refs["queue_table"] = ui.table(
            columns=[
                {"name": "job_id", "label": "Job ID", "field": "job_id", "align": "left"},
                {"name": "cycle_id", "label": "Cycle", "field": "cycle_id", "align": "left"},
                {"name": "created_at", "label": "Created At", "field": "created_at", "align": "left"},
                {"name": "summary", "label": "Summary", "field": "summary", "align": "left"},
            ],
            rows=[],
            row_key="job_id",
        ).classes("w-full")


def _build_tab_3_lab_journal(refs: dict[str, Any]) -> None:
    """Tab 3: Lab Journal - overlay plots, discrepancy/hypothesis text, Co-Pilot chat."""
    with ui.column().classes("w-full gap-4"):
        ui.label("Lab Journal (Scientific Hub)").classes("text-2xl font-bold")
        refs["journal_cycle_label"] = ui.label("Cycle: (none yet)").classes("text-sm text-gray-500")

        ui.label("Simulated vs. Actual Overlay").classes("text-lg font-semibold")
        refs["overlay_plot"] = ui.plotly(go.Figure()).classes("w-full")

        ui.separator()

        with ui.row().classes("w-full gap-4"):
            with ui.column().classes("flex-grow"):
                ui.label("Discrepancy Report (C_Analysis/discrepancy.md)").classes("font-semibold")
                refs["discrepancy_markdown"] = ui.markdown("*(no data yet)*").classes(
                    "w-full h-64 overflow-auto border rounded p-2"
                )
            with ui.column().classes("flex-grow"):
                ui.label("Hypothesis (C_Analysis/hypothesis.md)").classes("font-semibold")
                refs["hypothesis_markdown"] = ui.markdown("*(no data yet)*").classes(
                    "w-full h-64 overflow-auto border rounded p-2"
                )

        ui.separator()

        ui.label("🧑‍🔬 Co-Pilot Chat").classes("text-lg font-semibold")
        ui.label(
            "Notes submitted here are appended to this cycle's C_Analysis/copilot_feedback.md, "
            "which Gatekeeper reads on its next invocation."
        ).classes("text-sm text-gray-500")

        def send_copilot_note() -> None:
            message = refs["copilot_input"].value or ""
            if not message.strip():
                ui.notify("Enter a note before sending.", type="warning")
                return
            latest_cycle_dir = _find_latest_cycle_dir()
            if latest_cycle_dir is None:
                ui.notify("No active cycle yet - nothing to attach this note to.", type="warning")
                return
            _append_copilot_feedback(latest_cycle_dir, message)
            refs["copilot_input"].value = ""
            ui.notify(f"Note saved to {latest_cycle_dir.name}/C_Analysis/copilot_feedback.md", type="positive")

        with ui.row().classes("w-full items-end gap-2"):
            refs["copilot_input"] = ui.textarea(label="Co-Pilot note for the Gatekeeper").classes("flex-grow")
            ui.button("Send", on_click=send_copilot_note).props("icon=send")

        ui.label("Current copilot_feedback.md").classes("text-sm font-semibold mt-2")
        refs["copilot_feedback_markdown"] = ui.markdown("*(none yet)*").classes(
            "w-full h-40 overflow-auto border rounded p-2 text-sm"
        )


def _build_tab_4_settings(refs: dict[str, Any]) -> None:
    """Tab 4: System & LLM Settings - model assignment, hardware load, sampling sliders."""
    settings = _load_ui_settings()
    available_models = _list_available_ollama_models()

    with ui.column().classes("w-full gap-4"):
        ui.label("System & LLM Settings").classes("text-2xl font-bold")
        ui.label(
            "These controls persist to 00_System/ui_settings.json and are read dynamically by "
            "every agent (BaseAgent.__init__ for model assignment, BaseAgent.ask_llm for "
            "temperature/context size) - changes apply to the NEXT agent constructed or the "
            "NEXT ask_llm() call, not retroactively to work already in progress."
        ).classes("text-sm text-gray-500")

        ui.label("Per-Team Model Assignment").classes("text-lg font-semibold")
        refs["model_selects"] = {}
        with ui.grid(columns=2).classes("w-full gap-2"):
            for agent_name in ASSIGNABLE_AGENT_NAMES:
                current = settings["model_assignments"].get(agent_name, config.OLLAMA_MODEL)
                options = sorted(set(available_models) | {current})

                def make_handler(name: str):
                    def handler(event: Any) -> None:
                        current_settings = _load_ui_settings()
                        current_settings["model_assignments"][name] = event.value
                        _save_ui_settings(current_settings)

                    return handler

                select = ui.select(
                    options=options, value=current, label=agent_name, on_change=make_handler(agent_name)
                )
                refs["model_selects"][agent_name] = select

        ui.separator()

        ui.label("Hardware Load").classes("text-lg font-semibold")
        with ui.row().classes("w-full gap-6"):
            with ui.column().classes("flex-grow"):
                ui.label("CPU")
                refs["cpu_progress"] = ui.linear_progress(value=0.0, show_value=False)
                refs["cpu_label"] = ui.label("N/A")
            with ui.column().classes("flex-grow"):
                ui.label("RAM")
                refs["ram_progress"] = ui.linear_progress(value=0.0, show_value=False)
                refs["ram_label"] = ui.label("N/A")

        ui.separator()

        ui.label("Sampling Parameters").classes("text-lg font-semibold")

        def save_temperature(event: Any) -> None:
            current_settings = _load_ui_settings()
            current_settings["temperature"] = event.value
            _save_ui_settings(current_settings)

        def save_context_size(event: Any) -> None:
            current_settings = _load_ui_settings()
            current_settings["context_size"] = event.value
            _save_ui_settings(current_settings)

        ui.label("Temperature")
        ui.slider(min=0.0, max=1.0, step=0.05, value=settings["temperature"], on_change=save_temperature).props(
            "label-always"
        )

        ui.label("Context Size")
        ui.slider(
            min=512, max=32768, step=512, value=settings["context_size"], on_change=save_context_size
        ).props("label-always")


# --------------------------------------------------------------------------- #
# Refresh (the one and only ui.timer callback)
# --------------------------------------------------------------------------- #

def _refresh_tab_1(refs: dict[str, Any]) -> None:
    latest_cycle_dir = _find_latest_cycle_dir()

    # --- Definitive active-agent telemetry (Step 8) -----------------------------
    # This is the real, live signal - written by BaseAgent.execute() itself,
    # not inferred. Used both for its own direct readout AND to pick which
    # Kanban stage card shows as ACTIVE, in preference to the older
    # artifact-based inference (which now only serves as a fallback for the
    # brief window before any agent has ever run, or against a workspace
    # produced by a pre-Step-8 version of this codebase that never wrote
    # active_agent.json at all).
    live_agent_name, _live_status = _read_active_agent_status()
    refs["active_agent_label"].set_text(
        f"🟢 Currently Running: {live_agent_name}" if live_agent_name else "⚪ Idle (no agent currently executing)"
    )

    # Kanban stage strip
    stage_completion = _stage_completion(latest_cycle_dir) if latest_cycle_dir else []
    if live_agent_name is not None and live_agent_name in AGENT_NAME_TO_STAGE:
        active_stage = AGENT_NAME_TO_STAGE[live_agent_name]
    elif stage_completion:
        active_stage = _current_active_stage_name(stage_completion)
    else:
        active_stage = "Idle - No Cycle Yet"

    with refs["stage_row"]:
        for name, done in stage_completion:
            card = refs["stage_cards"].get(name)
            if card is None:
                with ui.card().classes("min-w-[10rem]") as card:
                    ui.label(name).classes("font-semibold")
                    status_label = ui.label("").classes("text-sm")
                    card.status_label = status_label  # type: ignore[attr-defined]
                refs["stage_cards"][name] = card

            is_active = name == active_stage
            card.classes(
                replace="min-w-[10rem] "
                + ("bg-blue-500 text-white" if is_active else ("bg-green-100" if done else "bg-gray-100"))
            )
            # When this stage IS the live-telemetry-confirmed active one, show
            # the exact agent name (e.g. "● ACTIVE (SandboxDebugger)") rather
            # than just the generic team-level "● ACTIVE" - a precision the
            # old, artifact-only inference could never offer since it had no
            # way to distinguish which agent WITHIN a multi-agent stage (e.g.
            # Team 3's PythonArchitect vs. SandboxDebugger) was currently running.
            if is_active and live_agent_name is not None and AGENT_NAME_TO_STAGE.get(live_agent_name) == name:
                status_text = f"● ACTIVE ({live_agent_name})"
            elif is_active:
                status_text = "● ACTIVE"
            else:
                status_text = "✓ done" if done else "pending"
            card.status_label.set_text(status_text)  # type: ignore[attr-defined]

    # E-Stop status
    active, _detail = _is_estop_active()
    refs["estop_status_label"].set_text(f"Status: {'🔴 TRIGGERED' if active else '🟢 Armed / Not Triggered'}")
    refs["estop_status_label"].classes(
        replace="text-sm " + ("text-red-700 font-bold" if active else "text-green-700")
    )

    # Summary feed
    summary_text = _safe_read_text(config.SUMMARY_FILE, default="*(summary.md is empty - no cycles completed yet.)*")
    refs["summary_markdown"].set_content(summary_text if summary_text.strip() else "*(summary.md is empty.)*")


def _refresh_tab_2(refs: dict[str, Any]) -> None:
    latest_cycle_dir = _find_latest_cycle_dir()
    state = _carousel_state(latest_cycle_dir)
    refs["carousel_state_label"].set_text(f"State: {state}")

    for i, card in enumerate(refs["station_cards"]):
        if state == "processing":
            card.classes(add="station-pulsing bg-blue-50")
            card.style(f"animation-delay: {i * 0.2}s")
        else:
            card.classes(remove="station-pulsing bg-blue-50")
            if state == "complete":
                card.classes(add="bg-green-50")
            else:
                card.classes(remove="bg-green-50")

    # Hardware queue table
    rows = []
    for entry in _safe_iterdir(config.HARDWARE_QUEUE_DIR):
        if entry.suffix != ".json":
            continue
        payload = _safe_read_json(entry)
        if not isinstance(payload, dict):
            continue
        station_4 = payload.get("station_4_cleanup", {})
        cleanup_text = station_4.get("cleanup_routine", "") if isinstance(station_4, dict) else ""
        rows.append(
            {
                "job_id": payload.get("job_id", entry.name),
                "cycle_id": payload.get("cycle_id", "?"),
                "created_at": payload.get("created_at", "?"),
                "summary": (cleanup_text[:60] if cleanup_text else "(no station_4_cleanup summary)"),
            }
        )
    refs["queue_table"].rows = rows
    refs["queue_table"].update()


def _refresh_tab_3(refs: dict[str, Any]) -> None:
    latest_cycle_dir = _find_latest_cycle_dir()
    if latest_cycle_dir is None:
        refs["journal_cycle_label"].set_text("Cycle: (none yet)")
        return

    refs["journal_cycle_label"].set_text(f"Cycle: {latest_cycle_dir.name}")
    refs["overlay_plot"].update_figure(_build_overlay_figure(latest_cycle_dir))

    discrepancy_text = _safe_read_text(
        latest_cycle_dir / "C_Analysis" / "discrepancy.md", default="*(discrepancy.md not written yet.)*"
    )
    refs["discrepancy_markdown"].set_content(discrepancy_text or "*(empty)*")

    hypothesis_text = _safe_read_text(
        latest_cycle_dir / "C_Analysis" / "hypothesis.md", default="*(hypothesis.md not written yet.)*"
    )
    refs["hypothesis_markdown"].set_content(hypothesis_text or "*(empty)*")

    copilot_text = _safe_read_text(
        latest_cycle_dir / "C_Analysis" / COPILOT_FEEDBACK_FILENAME, default="*(no Co-Pilot notes yet this cycle.)*"
    )
    refs["copilot_feedback_markdown"].set_content(copilot_text or "*(empty)*")


def _refresh_tab_4(refs: dict[str, Any]) -> None:
    cpu_percent, ram_percent = _read_hardware_load()
    if cpu_percent is not None:
        refs["cpu_progress"].set_value(cpu_percent / 100)
        refs["cpu_label"].set_text(f"{cpu_percent:.0f}%")
    else:
        refs["cpu_label"].set_text("N/A (psutil unavailable)")
    if ram_percent is not None:
        refs["ram_progress"].set_value(ram_percent / 100)
        refs["ram_label"].set_text(f"{ram_percent:.0f}%")
    else:
        refs["ram_label"].set_text("N/A (psutil unavailable)")


def _refresh_all(refs: dict[str, Any]) -> None:
    """The single ui.timer callback, fanning out to each tab's refresh logic."""
    try:
        _refresh_tab_1(refs)
        _refresh_tab_2(refs)
        _refresh_tab_3(refs)
        _refresh_tab_4(refs)
    except Exception:
        # A refresh tick failing must never take down the whole dashboard
        # process - log it and let the next poll (2s later) try again,
        # mirroring the same complete-fallback-safety philosophy used by
        # DeadlockManager elsewhere in this codebase.
        logger.exception("Dashboard refresh tick failed - will retry on the next poll.")


# --------------------------------------------------------------------------- #
# Page assembly & entry point
# --------------------------------------------------------------------------- #

@ui.page("/", title="Self-Driving Lab Dashboard")
def index() -> None:
    """
    The dashboard's single page. Decorated with `@ui.page('/')` (rather
    than left as bare top-level `ui.*` calls) SPECIFICALLY so this function
    runs FRESH FOR EVERY CLIENT CONNECTION - each browser tab that opens the
    dashboard gets its own independent `refs` dict and its own `ui.timer`
    closure over that dict.

    This matters correctness-wise: without an explicit page registration,
    NiceGUI still runs top-level UI-building code once per client
    connection (there is no way to opt out of that), but a MODULE-LEVEL
    `refs` dict would then be silently overwritten every time a new client
    connects - the first client's `ui.timer` callback would keep firing,
    but against elements that now belong to a DIFFERENT client's page,
    effectively updating the wrong browser tab. Declaring `refs` as a local
    variable HERE guarantees each client's timer closure captures its own
    isolated dict, so multiple simultaneous dashboard viewers (e.g. two lab
    members with the page open at once) each see their own page update
    correctly and independently.
    """
    refs: dict[str, Any] = {}

    with ui.tabs().classes("w-full") as tabs:
        tab1 = ui.tab("Mission Control")
        tab2 = ui.tab("Hardware Live-Monitor")
        tab3 = ui.tab("Lab Journal")
        tab4 = ui.tab("System & LLM Settings")

    with ui.tab_panels(tabs, value=tab1).classes("w-full"):
        with ui.tab_panel(tab1):
            _build_tab_1_mission_control(refs)
        with ui.tab_panel(tab2):
            _build_tab_2_hardware_monitor(refs)
        with ui.tab_panel(tab3):
            _build_tab_3_lab_journal(refs)
        with ui.tab_panel(tab4):
            _build_tab_4_settings(refs)

    ui.timer(POLL_INTERVAL_SECONDS, lambda: _refresh_all(refs))
    _refresh_all(refs)  # populate immediately on load, rather than waiting for the first tick

# NiceGUI's documented multiprocessing-reload guard: `ui.run()` must only be
# called under this exact condition, since NiceGUI spawns a subprocess
# (`__mp_main__`) internally when auto-reload is enabled.
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Self-Driving Lab Dashboard",
        port=int(os.getenv("UI_DASHBOARD_PORT", "8080")),
        reload=False,
        show=False,
    )
