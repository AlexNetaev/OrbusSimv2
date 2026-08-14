"""
src package for the Self-Driving Lab autonomous multi-agent system.

This package contains the shared "infrastructure" code used by every agent:
- config.py            : central configuration (paths, model names, constants)
- workspace_manager.py  : creates / maintains the on-disk workspace agents communicate through
- llm_wrapper.py        : robust wrapper around the Ollama client with Pydantic-validated
                           structured outputs and auto-correction retries

Because agents in this system are STATELESS and communicate purely by reading and writing
files in the workspace, this package intentionally has no hidden global state beyond the
Path constants in config.py. Every function here is safe to call from any agent process,
at any time, in any order (idempotent where it matters).
"""

__version__ = "0.1.0"
