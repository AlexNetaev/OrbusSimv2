"""
config.py
=========
Single source of truth for filesystem paths and model configuration.

Design rationale:
------------------
Because this system is a *file-system-based* multi-agent architecture (agents are
stateless processes that read/write Markdown & JSON to a shared workspace), it is
critical that every agent process resolves paths identically. If each module computed
its own relative paths, agents launched from different working directories would write
to different physical locations and silently desync the "shared memory" of the system.

To prevent that class of bug, ALL paths are:
  1. Computed once, here, as absolute `pathlib.Path` objects.
  2. Anchored off this file's location (not off `os.getcwd()`), so it doesn't matter
     where main_loop.py or an agent script is actually invoked from.
  3. Exposed as module-level constants that other modules import rather than re-derive.

Environment overrides:
-----------------------
We use python-dotenv so deployment-specific values (custom workspace root, alternate
Ollama host, alternate default model) can be supplied via a `.env` file without touching
code.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# Load environment variables from a .env file in the project root, if present.
# This is a no-op (and perfectly safe) if no .env file exists.
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------------
# config.py lives at <project_root>/src/config.py, so the project root is two
# levels up from this file (src/ -> project_root/).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# WORKSPACE ROOT
# ---------------------------------------------------------------------------
# The workspace is the "shared brain" all agents read/write through. It can be
# relocated (e.g. to a mounted volume in production) via the WORKSPACE_ROOT
# environment variable; otherwise it defaults to <project_root>/workspace.
WORKSPACE_ROOT: Path = Path(os.getenv("WORKSPACE_ROOT", PROJECT_ROOT / "workspace")).resolve()

# ---------------------------------------------------------------------------
# WORKSPACE SUBDIRECTORIES
# ---------------------------------------------------------------------------
# Named exactly per the required layout. Numeric prefixes keep them in a sane,
# human-readable order when browsed in a file explorer or `ls`.
SYSTEM_DIR: Path = WORKSPACE_ROOT / "00_System"
KNOWLEDGE_BASE_DIR: Path = WORKSPACE_ROOT / "01_Knowledge_Base"
KNOWLEDGE_BASE_ARCHIVE_DIR: Path = KNOWLEDGE_BASE_DIR / "Archive"
RESEARCH_CYCLES_DIR: Path = WORKSPACE_ROOT / "02_Research_Cycles"
HARDWARE_QUEUE_DIR: Path = WORKSPACE_ROOT / "03_Hardware_Queue"

# ---------------------------------------------------------------------------
# WELL-KNOWN FILES
# ---------------------------------------------------------------------------
# These are the "bootstrap" files every agent expects to exist (even if empty)
# so that read operations never need to special-case a missing file.
DIRECTIVE_FILE: Path = SYSTEM_DIR / "directive.md"
HARDWARE_LIMITS_FILE: Path = SYSTEM_DIR / "hardware_limits.yaml"
SUMMARY_FILE: Path = SYSTEM_DIR / "summary.md"
THEORY_BASELINE_FILE: Path = KNOWLEDGE_BASE_DIR / "theory_baseline.md"

# ---------------------------------------------------------------------------
# UI-BACKEND INTEGRATION FILES (Step 8)
# ---------------------------------------------------------------------------
# These three files are the shared contract between src/ui_dashboard.py (a
# separate process) and main_loop.py / src/base_agent.py. Centralizing their
# paths here - rather than each side independently reconstructing
# `SYSTEM_DIR / "some_filename.json"` - is what guarantees the writer
# (ui_dashboard.py's E-Stop button, Tab 4 settings) and the readers
# (main_loop.py's E-Stop check, BaseAgent's model/temperature resolution)
# can never silently drift onto different physical paths.

# Written by ui_dashboard.py's Tab 1 "Global Software E-Stop" button;
# checked by main_loop.py's run_cycle_sequence() at the start and between
# every team boundary (see EmergencyStopException).
ESTOP_FLAG_FILE: Path = SYSTEM_DIR / "ESTOP.flag"

# Written by ui_dashboard.py's Tab 4 (per-team model assignment, temperature,
# context size); read by BaseAgent as an optional override layer on top of
# the OLLAMA_MODEL/DEFAULT_TEMPERATURE env-var defaults below. Entirely
# optional - every agent works identically to before if this file has never
# been written (i.e. the dashboard has never been run).
UI_SETTINGS_FILE: Path = SYSTEM_DIR / "ui_settings.json"

# Written by BaseAgent.execute() at the start and end (finally block) of
# every single agent invocation; read by ui_dashboard.py's Mission Control
# tab for a definitive, real-time "which agent is running right now"
# readout, replacing the earlier artifact-based inference.
ACTIVE_AGENT_STATUS_FILE: Path = SYSTEM_DIR / "active_agent.json"

# ---------------------------------------------------------------------------
# LLM / OLLAMA CONFIGURATION
# ---------------------------------------------------------------------------
# Default model used across all agents unless an agent explicitly overrides it.
# Can be overridden per-deployment via the OLLAMA_MODEL env var.
# OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

# Ollama server address. The `ollama` python client defaults to localhost:11434
# if OLLAMA_HOST isn't set, but we surface it here so it's visible/overridable
# in one place.
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Default sampling temperature for structured-output tasks. Kept low because
# strict JSON generation benefits from low-variance, deterministic-leaning output.
DEFAULT_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

# Default max retries for LLM calls that require validated/structured output.
DEFAULT_MAX_RETRIES: int = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

# Default context window size (Ollama's `num_ctx` option), used whenever
# neither an explicit ask_llm() call nor UI_SETTINGS_FILE specifies one.
# Matches src/ui_dashboard.py's Tab 4 slider default, so a dashboard that's
# never been touched still reports the value agents are actually using.
DEFAULT_CONTEXT_SIZE: int = int(os.getenv("OLLAMA_CONTEXT_SIZE", "4096"))

# Konfigurierbarer Timeout für wait_for_hardware(), damit das System
# nicht unendlich hängt, wenn der externe Simulator nie schreibt.
HARDWARE_WAIT_TIMEOUT_SECONDS: int = int(os.getenv("HARDWARE_WAIT_TIMEOUT", "600"))

def all_workspace_dirs() -> list[Path]:
    """
    Return every directory that must exist for the workspace to be considered
    "initialized". Centralized here so WorkspaceManager and any future tooling
    (health checks, tests, etc.) share one definition instead of duplicating it.
    """
    return [
        WORKSPACE_ROOT,
        SYSTEM_DIR,
        KNOWLEDGE_BASE_DIR,
        KNOWLEDGE_BASE_ARCHIVE_DIR,
        RESEARCH_CYCLES_DIR,
        HARDWARE_QUEUE_DIR,
    ]
