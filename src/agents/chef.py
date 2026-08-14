"""
chef.py
========
The Chef Agent - master coordinator of the Self-Driving Lab.

Role in the pipeline:
-----------------------
The Chef runs LAST in every research cycle, after every other agent (Planner,
Researcher, HardwareExecutor, DataAnalyst) has done its work. Its job is
purely synthesis and judgment, not experimentation:

  1. Read the lab's overarching goal (`00_System/directive.md`).
  2. Read the running history of what's happened so far
     (`00_System/summary.md`).
  3. Read what THIS cycle specifically produced - primarily the DataAnalyst's
     `discrepancy.md` reality check, plus anything else living in
     `cycle_dir/C_Analysis/`.
  4. Decide: has the directive been satisfied? If not, what should the next
     cycle focus on?
  5. Append a strictly-bounded 3-line summary of this cycle to the GLOBAL
     summary log, and archive its full reasoning to shadow memory.

Why the Chef enforces a hard 3-line-per-cycle summary limit:
-----------------------------------------------------------------
`00_System/summary.md` is the ONE piece of "long-term memory" every future
agent in every future cycle will read in full, from the very first cycle
onward. If each cycle appended an unbounded amount of text, summary.md would
grow linearly with the number of cycles and eventually blow the context
window of every agent that reads it (in this run, and every agent afterward,
forever). Capping each cycle's contribution to at most 3 concise lines is a
deliberate token-hygiene policy, enforced HERE in code (not just requested
via prompt) - the LLM might comply, but code truncation is what actually
guarantees the invariant regardless of the model's behavior.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src import config
from src.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Hard cap on how many summary lines the Chef may append to summary.md per
# cycle. Enforced both in the Pydantic model (via a validator, so a
# non-conforming LLM response fails validation and triggers the
# auto-correction retry loop in llm_wrapper.py) AND again defensively at
# write-time (see _append_to_global_summary), so the invariant holds even if
# a future code change accidentally bypasses the model validation path.
MAX_SUMMARY_LINES_PER_CYCLE = 3

# If a discrepancy.md (or other analysis artifact) is enormous, we don't want
# to blow the Chef's prompt budget reading it verbatim. This caps how many
# characters of each C_Analysis file we forward into the prompt.
MAX_ANALYSIS_FILE_CHARS = 4000


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class ChefDecisionModel(BaseModel):
    """
    Strict output schema the LLM must populate. This is the Chef's entire
    judgment for the cycle: whether the lab's directive is satisfied, a
    token-bounded summary for the permanent record, and guidance for
    whatever comes next.
    """

    goal_achieved: bool = Field(
        ...,
        description="Whether the objective described in directive.md is now fully satisfied.",
    )
    summary_lines: list[str] = Field(
        ...,
        description=(
            f"Exactly {MAX_SUMMARY_LINES_PER_CYCLE} concise lines summarizing this cycle's "
            "progress and outcome, suitable for a permanent, space-constrained global log."
        ),
    )
    next_recommendation: str = Field(
        ...,
        description=(
            "Guidance for the next cycle (what to try, adjust, or investigate), OR, if "
            "goal_achieved is True, a final concluding statement for the project."
        ),
    )

    @field_validator("summary_lines")
    @classmethod
    def _enforce_exact_line_count(cls, lines: list[str]) -> list[str]:
        """
        Reject (rather than silently truncate/pad) a response that doesn't
        contain exactly MAX_SUMMARY_LINES_PER_CYCLE lines. Raising here
        surfaces a proper ValidationError, which llm_wrapper.py's
        auto-correction loop will catch and feed back to the model as a
        concrete error to fix - e.g. "give me 3 lines, not 5" - rather than
        us quietly reshaping whatever the model gave us and potentially
        losing information the model considered important.
        """
        if len(lines) != MAX_SUMMARY_LINES_PER_CYCLE:
            raise ValueError(
                f"summary_lines must contain exactly {MAX_SUMMARY_LINES_PER_CYCLE} lines, "
                f"got {len(lines)}."
            )
        for line in lines:
            if not line.strip():
                raise ValueError("summary_lines must not contain empty/whitespace-only lines.")
        return lines


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

class Chef(BaseAgent):
    """
    Reads the global directive, global summary, and this cycle's analysis
    output; asks the LLM for a structured go/no-go decision; appends a
    bounded summary to the permanent record; and archives full reasoning to
    shadow memory.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        # Fixed agent name per spec - "Chef" is this agent's identity for
        # logging/shadow-memory purposes system-wide, not configurable.
        super().__init__(agent_name="Chef", workspace_path=workspace_path, model=model)

        # goal_achieved is surfaced as an instance attribute (in addition to
        # being written to disk) so that main_loop.py can inspect
        # `chef.goal_achieved` immediately after `chef.execute(...)` without
        # needing to re-read/re-parse shadow memory from disk. This is a
        # convenience for the SAME process/run only - because agents are
        # otherwise stateless, nothing else in the system should ever rely
        # on this attribute surviving across separate Chef instantiations.
        self.goal_achieved: bool = False

    # ------------------------------------------------------------------ #
    # BaseAgent contract
    # ------------------------------------------------------------------ #

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read directive.md + summary.md (global, workspace-level state).
            2. Read this cycle's C_Analysis/ output (cycle-local state).
            3. Ask the LLM for a ChefDecisionModel.
            4. Append the (validated, exactly-3-line) summary to summary.md.
            5. Write the full reasoning + payload to
               cycle_dir/D_Shadow_Memory/chef_shadow.json.
        """
        directive_text = self._read_text_or_placeholder(
            config.DIRECTIVE_FILE, placeholder="(directive.md is empty - no directive has been set yet.)"
        )
        summary_text = self._read_text_or_placeholder(
            config.SUMMARY_FILE, placeholder="(summary.md is empty - this is the first cycle.)"
        )
        analysis_bundle = self._read_cycle_analysis(cycle_dir)

        prompt = self._build_decision_prompt(
            directive_text=directive_text,
            summary_text=summary_text,
            analysis_bundle=analysis_bundle,
        )
        system_prompt = (
            "You are the Chef: the master coordinator of an autonomous self-driving "
            "laboratory. You do not run experiments yourself - you synthesize what has "
            "already happened and make the final call for this cycle. Be precise and "
            "conservative: only declare goal_achieved=true if the evidence in this "
            "cycle's analysis genuinely and fully satisfies the directive. Your "
            "summary_lines become a PERMANENT, space-constrained record that every "
            "future cycle will read - make every line count."
        )

        decision = self.ask_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            response_model=ChefDecisionModel,
        )
        # `ask_llm` is typed to return `str | T`; when response_model is given
        # it always returns a validated T on success (or raises otherwise), so
        # this assertion is purely a static-typing aid, not a real runtime risk.
        assert isinstance(decision, ChefDecisionModel)

        # Surface the decision on the instance for main_loop.py to inspect
        # immediately after execute() returns (see __init__ docstring above).
        self.goal_achieved = decision.goal_achieved

        self._append_to_global_summary(cycle_dir=cycle_dir, decision=decision)
        self._write_shadow_memory(
            cycle_dir=cycle_dir,
            system_prompt=system_prompt,
            prompt=prompt,
            directive_text=directive_text,
            analysis_bundle=analysis_bundle,
            decision=decision,
        )

        if decision.goal_achieved:
            self.logger.info(
                "[Chef] *** MILESTONE: directive.md goal reported ACHIEVED at cycle %s. ***",
                cycle_dir.name,
            )
        self.logger.info(
            "[Chef] Decision complete | goal_achieved=%s | next_recommendation=%r",
            decision.goal_achieved,
            decision.next_recommendation,
        )

    # ------------------------------------------------------------------ #
    # Reading global + cycle-local state
    # ------------------------------------------------------------------ #

    def _read_text_or_placeholder(self, path: Path, placeholder: str) -> str:
        """
        Read a file's text content, or return a descriptive placeholder if
        the file doesn't exist or is empty.

        Both `directive.md` and `summary.md` are guaranteed to EXIST (as
        empty files) by WorkspaceManager, but "exists and is empty" is a
        completely normal, expected state early in the project's life (no
        directive set yet, or this is cycle #1 with no history yet). We
        don't want the LLM prompt to just show an empty string in that case -
        an explicit placeholder is much clearer context for the model than
        silence, which it might otherwise misinterpret as a formatting error.
        """
        if not path.exists():
            self.logger.warning("[Chef] Expected file missing: %s. Using placeholder text.", path)
            return placeholder

        text = path.read_text(encoding="utf-8").strip()
        return text if text else placeholder

    def _read_cycle_analysis(self, cycle_dir: Path) -> dict[str, str]:
        """
        Read every file in `cycle_dir/C_Analysis/` (the DataAnalyst's
        discrepancy.md and any other analysis artifacts a future agent might
        add there) into a {filename: contents} dict, each capped at
        MAX_ANALYSIS_FILE_CHARS to keep the Chef's prompt bounded regardless
        of how verbose an upstream agent's output gets.

        We intentionally read the whole directory rather than hardcoding
        just "discrepancy.md" - the spec calls out discrepancy.md specifically
        but also "any analysis outputs in cycle_dir/C_Analysis/", and reading
        the directory generically means this agent doesn't need to change if
        a future agent adds e.g. a second analysis file alongside it.
        """
        analysis_dir = cycle_dir / "C_Analysis"
        if not analysis_dir.exists():
            self.logger.warning(
                "[Chef] C_Analysis directory not found at %s - proceeding with no analysis context.",
                analysis_dir,
            )
            return {}

        analysis_files: dict[str, str] = {}
        # Sorted for deterministic prompt ordering across runs (helps
        # reproducibility when debugging/comparing shadow memory later).
        for file_path in sorted(analysis_dir.iterdir()):
            if not file_path.is_file():
                continue
            content = file_path.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue
            if len(content) > MAX_ANALYSIS_FILE_CHARS:
                omitted = len(content) - MAX_ANALYSIS_FILE_CHARS
                content = content[:MAX_ANALYSIS_FILE_CHARS] + f"\n... ({omitted} further character(s) omitted) ..."
            analysis_files[file_path.name] = content

        if not analysis_files:
            self.logger.warning(
                "[Chef] C_Analysis directory at %s contained no readable files - "
                "proceeding with no analysis context.",
                analysis_dir,
            )

        return analysis_files

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _build_decision_prompt(
        self,
        directive_text: str,
        summary_text: str,
        analysis_bundle: dict[str, str],
    ) -> str:
        """
        Assemble the full prompt: the global directive, the running summary
        history, and this cycle's analysis output, then ask for the
        structured decision.
        """
        if analysis_bundle:
            analysis_block = "\n\n".join(
                f"### {filename}\n```\n{content}\n```" for filename, content in analysis_bundle.items()
            )
        else:
            analysis_block = "(No analysis files were found for this cycle.)"

        return f"""Review the lab's directive, its history so far, and this cycle's results, then make your decision.

## Global Directive (directive.md)
```
{directive_text}
```

## Global Summary Log So Far (summary.md)
```
{summary_text}
```

## This Cycle's Analysis Output (C_Analysis/)
{analysis_block}

## Your task
Based on ALL of the above, determine:
1. `goal_achieved`: Is the objective in the Global Directive now fully satisfied by the
   evidence available (this cycle's analysis, combined with the history in the summary
   log)? Be conservative - only true if genuinely and fully satisfied.
2. `summary_lines`: EXACTLY {MAX_SUMMARY_LINES_PER_CYCLE} concise, non-empty lines
   summarizing this cycle's progress and outcome. These lines get permanently appended
   to the global summary log that every future cycle will read - be specific and dense,
   avoid filler.
3. `next_recommendation`: If goal_achieved is False, concrete guidance for what the next
   cycle should focus on. If goal_achieved is True, a final concluding statement for the
   project.
"""

    # ------------------------------------------------------------------ #
    # Global memory update
    # ------------------------------------------------------------------ #

    def _append_to_global_summary(self, cycle_dir: Path, decision: ChefDecisionModel) -> None:
        """
        Append this cycle's bounded summary to the PERMANENT, global
        `00_System/summary.md`.

        Token-hygiene enforcement:
            `ChefDecisionModel._enforce_exact_line_count` already rejects any
            LLM response that doesn't contain exactly
            MAX_SUMMARY_LINES_PER_CYCLE lines (triggering the auto-correction
            retry loop in llm_wrapper.py). As defense in depth, we ALSO
            truncate to the first MAX_SUMMARY_LINES_PER_CYCLE lines again
            here at write-time - this is what actually guarantees the
            invariant holds on disk even if the Pydantic validator is ever
            weakened or bypassed in a future refactor.

        Each cycle's contribution is written as a clearly delimited,
        cycle-labeled block (rather than bare, unattributed lines) so a
        human or agent reading summary.md later can always tell which cycle
        each line came from.
        """
        bounded_lines = decision.summary_lines[:MAX_SUMMARY_LINES_PER_CYCLE]

        entry_lines = [f"## {cycle_dir.name}"]
        entry_lines.extend(f"- {line.strip()}" for line in bounded_lines)
        entry_block = "\n".join(entry_lines) + "\n"

        # Append rather than overwrite: summary.md is the permanent record of
        # every cycle that has ever run. Open in append mode with an
        # explicit leading blank line so each cycle's block is visually
        # separated regardless of what was already in the file (including
        # the placeholder empty-file case from WorkspaceManager).
        with config.SUMMARY_FILE.open("a", encoding="utf-8") as f:
            f.write("\n" + entry_block)

        self.logger.info(
            "[Chef] Appended %d summary line(s) for %s to %s",
            len(bounded_lines),
            cycle_dir.name,
            config.SUMMARY_FILE,
        )

    # ------------------------------------------------------------------ #
    # Shadow memory persistence
    # ------------------------------------------------------------------ #

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        system_prompt: str,
        prompt: str,
        directive_text: str,
        analysis_bundle: dict[str, str],
        decision: ChefDecisionModel,
    ) -> None:
        """
        Persist the full LLM exchange and inputs as JSON, separate from the
        (append-only, token-bounded) global summary.md. This is the Chef's
        complete audit trail for this cycle - useful for debugging why a
        particular goal_achieved verdict was reached, without needing to
        re-run the LLM or reconstruct the prompt from scratch.
        """
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "agent_name": self.agent_name,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "inputs": {
                "directive_text": directive_text,
                "analysis_files_read": list(analysis_bundle.keys()),
            },
            "validated_response": decision.model_dump(),
        }

        shadow_path = shadow_dir / "chef_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[Chef] Wrote shadow memory to %s", shadow_path)
