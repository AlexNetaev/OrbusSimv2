"""
knowledge_curator.py
======================
Team 2 - Knowledge Curator Agent ("KnowledgeCurator").

Role in the pipeline:
-----------------------
This agent runs immediately after HypothesisArchitect and right before Chef.
Where HypothesisArchitect explains ONE cycle's outcome, KnowledgeCurator's job
is to fold that single-cycle insight into the lab's PERMANENT, cross-cycle
scientific understanding:

  1. Read the current global theory baseline
     (`01_Knowledge_Base/theory_baseline.md`) - the lab's standing
     understanding of the underlying chemistry/physics, accumulated across
     every prior cycle.
  2. Read this cycle's fresh hypothesis
     (`cycle_dir/C_Analysis/hypothesis.md`, written by HypothesisArchitect).
  3. Ask the LLM to produce a CONSOLIDATED replacement baseline that folds in
     whatever this cycle's hypothesis adds or corrects - not just appends the
     new finding, but genuinely merges it into a coherent, non-redundant
     document.
  4. Overwrite `theory_baseline.md` with that consolidated version.
  5. Move anything the LLM identifies as now-outdated/superseded/redundant
     OUT of the active baseline and into a permanent, timestamped archive
     (`01_Knowledge_Base/Archive/theory_archive.md`), so nothing is ever
     silently lost - it's relocated, not deleted.

Why token hygiene matters here more than almost anywhere else in the system:
-------------------------------------------------------------------------------
`theory_baseline.md` is read in FULL by MachinePlanner and HypothesisArchitect
on every single future cycle, forever. Unlike `summary.md` (which Chef caps at
a hard 3 lines per cycle by construction), the theory baseline's job is to
represent current understanding as a coherent whole - it can't simply be
truncated to N lines without risking losing scientifically load-bearing
content. So this agent can't enforce a hard cap the way Chef does; instead it:
  - Actively asks the LLM to CONSOLIDATE (merge, de-duplicate, prune) rather
    than just append, keeping growth sub-linear in the common case.
  - Provides an explicit archiving pathway so genuinely superseded content has
    somewhere to go OTHER than bloating the active file.
  - Loudly logs a warning (not a hard failure - a working baseline that's
    slightly over budget is still better than crashing the pipeline) if the
    result still exceeds `MAX_BASELINE_CHARS`, so a human maintaining the lab
    has clear, actionable signal that the next cycle's KnowledgeCurator (or a
    human) needs to prune harder.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from src import config
from src.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Soft ceiling on theory_baseline.md's size, in characters. This is NOT
# enforced by truncating the file (that would risk silently destroying
# scientifically load-bearing content) - it's a monitoring threshold: if the
# LLM's consolidated output still exceeds it, we log a loud warning so a
# human (or a future, smarter consolidation prompt) knows more aggressive
# pruning is needed. ~4000 characters is a deliberately conservative budget
# relative to typical local-LLM context windows, leaving ample headroom for
# the rest of each prompt (directive, hardware limits, discrepancy report, etc.)
# that gets read alongside it.
MAX_BASELINE_CHARS = 4000

# Standard header written into theory_baseline.md the first time this agent
# ever encounters it empty/missing, so the file always has a consistent,
# parseable shape from cycle 1 onward rather than starting as a bare blob of
# LLM-generated prose with no structure.
STANDARD_BASELINE_HEADER = "# Theory Baseline\n\n*(No findings recorded yet.)*\n"

# Filename HypothesisArchitect is expected to have written into
# cycle_dir/C_Analysis/.
HYPOTHESIS_REPORT_FILENAME = "hypothesis.md"

# Filename this agent appends archived/superseded content to, under
# config.KNOWLEDGE_BASE_ARCHIVE_DIR.
THEORY_ARCHIVE_FILENAME = "theory_archive.md"


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class TheoryUpdateModel(BaseModel):
    """
    Strict output schema the LLM must populate: the full, consolidated
    replacement theory baseline, plus anything that should be pruned out of
    it into the permanent archive.
    """

    updated_theory_baseline: str = Field(
        ...,
        description=(
            "The complete, new, consolidated version of theory_baseline.md, incorporating "
            "this cycle's hypothesis. This REPLACES the entire file - it must be a full, "
            "coherent document on its own, not just the new addition."
        ),
    )
    archived_knowledge: str | None = Field(
        default=None,
        description=(
            "Any outdated, superseded, or redundant hypotheses/theories that should be "
            "pruned OUT of the active baseline as a result of this update. None or an empty "
            "string if nothing needs to be archived this cycle."
        ),
    )


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

class KnowledgeCurator(BaseAgent):
    """
    Reads the global theory baseline and this cycle's hypothesis; asks the
    LLM to produce a consolidated replacement baseline plus anything that
    should be archived; writes the new baseline, archives superseded
    content with a timestamp, monitors token hygiene, and persists the full
    audit trail to shadow memory.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        # Fixed agent name per spec - "KnowledgeCurator" is this agent's
        # identity for logging/shadow-memory purposes system-wide.
        super().__init__(agent_name="KnowledgeCurator", workspace_path=workspace_path, model=model)

    # ------------------------------------------------------------------ #
    # BaseAgent contract
    # ------------------------------------------------------------------ #

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read theory_baseline.md (writing a standard header first if
               it's missing or empty) and this cycle's hypothesis.md.
            2. Ask the LLM for a TheoryUpdateModel: a full consolidated
               replacement baseline, plus anything to archive.
            3. Overwrite theory_baseline.md with the consolidated version.
            4. If any content was flagged for archiving, append it (with a
               timestamp and cycle reference) to Archive/theory_archive.md.
            5. Check the new baseline's size against MAX_BASELINE_CHARS and
               log a loud warning if it's still over budget.
            6. Write the full prompt/response audit trail to
               cycle_dir/D_Shadow_Memory/knowledge_curator_shadow.json.
        """
        current_baseline_text = self._read_or_initialize_baseline()
        hypothesis_text = self._read_hypothesis_report(cycle_dir)

        prompt = self._build_consolidation_prompt(
            current_baseline_text=current_baseline_text,
            hypothesis_text=hypothesis_text,
        )
        system_prompt = (
            "You are the Knowledge Curator for an autonomous self-driving laboratory. "
            "You maintain the lab's single, permanent theory baseline document - the "
            "consolidated scientific understanding that every future cycle's planning and "
            "analysis will rely on. When incorporating a new finding, MERGE it into the "
            "existing document: update or replace any statement it supersedes, remove "
            "redundancy, and keep the result coherent and concise. Never simply append the "
            "new finding on top of the old text - that produces an ever-growing, "
            "increasingly redundant document, which is exactly what you must prevent. If "
            "nothing in the current baseline is actually outdated by this cycle's finding, "
            "leave it in place and only add what's genuinely new."
        )

        theory_update = self.ask_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            response_model=TheoryUpdateModel,
        )
        # `ask_llm` is typed to return `str | T`; when response_model is given
        # it always returns a validated T on success (or raises otherwise), so
        # this assertion is purely a static-typing aid, not a real runtime risk.
        assert isinstance(theory_update, TheoryUpdateModel)

        self._write_updated_baseline(theory_update.updated_theory_baseline)
        self._archive_superseded_knowledge(cycle_dir=cycle_dir, archived_knowledge=theory_update.archived_knowledge)
        self._check_token_hygiene(theory_update.updated_theory_baseline)

        self._write_shadow_memory(
            cycle_dir=cycle_dir,
            system_prompt=system_prompt,
            prompt=prompt,
            previous_baseline_text=current_baseline_text,
            hypothesis_text=hypothesis_text,
            theory_update=theory_update,
        )

        self.logger.info(
            "[KnowledgeCurator] Baseline updated | new_length=%d chars | archived=%s",
            len(theory_update.updated_theory_baseline),
            bool(theory_update.archived_knowledge and theory_update.archived_knowledge.strip()),
        )

    # ------------------------------------------------------------------ #
    # Reading inputs
    # ------------------------------------------------------------------ #

    def _read_or_initialize_baseline(self) -> str:
        """
        Read `theory_baseline.md`. If it doesn't exist or is empty (the
        normal state for a brand-new workspace, or the very first cycle),
        write the standard header into it FIRST, then return that header as
        the "current" text.

        Writing the header immediately (rather than only using it as an
        in-memory placeholder, the way `Chef`/`MachinePlanner` handle their
        placeholder text) matters here specifically because THIS agent's job
        is to own and maintain this exact file - it should never leave it in
        a genuinely empty/uninitialized state once it has run, even if the
        very first LLM call were to somehow fail before producing a proper
        update.
        """
        if not config.THEORY_BASELINE_FILE.exists():
            self.logger.warning(
                "[KnowledgeCurator] theory_baseline.md not found at %s - creating with standard header.",
                config.THEORY_BASELINE_FILE,
            )
            config.THEORY_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
            config.THEORY_BASELINE_FILE.write_text(STANDARD_BASELINE_HEADER, encoding="utf-8")
            return STANDARD_BASELINE_HEADER

        text = config.THEORY_BASELINE_FILE.read_text(encoding="utf-8")
        if not text.strip():
            self.logger.warning(
                "[KnowledgeCurator] theory_baseline.md at %s is empty - initializing with standard header.",
                config.THEORY_BASELINE_FILE,
            )
            config.THEORY_BASELINE_FILE.write_text(STANDARD_BASELINE_HEADER, encoding="utf-8")
            return STANDARD_BASELINE_HEADER

        return text

    def _read_hypothesis_report(self, cycle_dir: Path) -> str:
        """
        Read this cycle's `hypothesis.md` (written by HypothesisArchitect).

        Handled gracefully if missing, per spec: rather than raising (which
        would abort the whole cycle), we log a warning and return an
        explicit placeholder. A missing hypothesis is still meaningful input
        for the consolidation prompt below - the correct LLM behavior in
        that case is simply "no new finding to incorporate this cycle,
        return the baseline unchanged" - exactly analogous to how
        HypothesisArchitect itself handles a missing discrepancy.md.
        """
        hypothesis_path = cycle_dir / "C_Analysis" / HYPOTHESIS_REPORT_FILENAME
        if not hypothesis_path.exists():
            self.logger.warning(
                "[KnowledgeCurator] hypothesis.md not found at %s - proceeding with no new "
                "finding to incorporate this cycle.",
                hypothesis_path,
            )
            return (
                "(hypothesis.md not found for this cycle - the Hypothesis Architect may not "
                "have run yet, or produced no output. No new finding to incorporate.)"
            )

        text = hypothesis_path.read_text(encoding="utf-8").strip()
        if not text:
            self.logger.warning(
                "[KnowledgeCurator] hypothesis.md at %s is empty - proceeding with no new "
                "finding to incorporate this cycle.",
                hypothesis_path,
            )
            return "(hypothesis.md is empty for this cycle. No new finding to incorporate.)"

        return text

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _build_consolidation_prompt(self, current_baseline_text: str, hypothesis_text: str) -> str:
        """
        Assemble the full prompt: the current baseline + this cycle's fresh
        hypothesis, then ask for the consolidated replacement.
        """
        return f"""Consolidate this cycle's new finding into the lab's permanent theory baseline.

## Current Theory Baseline (theory_baseline.md)
```
{current_baseline_text}
```

## This Cycle's New Hypothesis (C_Analysis/hypothesis.md)
```
{hypothesis_text}
```

## Your task
Produce an updated theory baseline by determine:
1. `updated_theory_baseline`: The COMPLETE, new version of theory_baseline.md. Merge this
   cycle's finding into the existing document rather than appending it - update or replace
   any existing statement the new finding supersedes, remove redundancy, and keep the
   result coherent, well-organized, and as CONCISE as possible while losing no
   scientifically load-bearing content. This is a full replacement document, not a diff.
2. `archived_knowledge`: Any specific statement(s) from the CURRENT baseline that are now
   outdated, superseded, or redundant as a result of this update, and should be removed
   from the active document and preserved in the archive instead. Return null/empty if
   nothing needs to be archived this cycle (e.g. the new finding is purely additive with no
   contradiction of prior theory).
"""

    # ------------------------------------------------------------------ #
    # Writing the updated baseline
    # ------------------------------------------------------------------ #

    def _write_updated_baseline(self, updated_theory_baseline: str) -> None:
        """
        Overwrite `theory_baseline.md` with the LLM's consolidated version.

        This is a full REPLACEMENT (not an append) by design: the whole
        point of consolidation is that the new file already contains
        everything from the old file that's still valid, merged with
        whatever's new. Appending on top would defeat that purpose entirely
        and guarantee unbounded growth.
        """
        config.THEORY_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.THEORY_BASELINE_FILE.write_text(updated_theory_baseline, encoding="utf-8")
        self.logger.info(
            "[KnowledgeCurator] Wrote updated theory baseline (%d chars) to %s",
            len(updated_theory_baseline),
            config.THEORY_BASELINE_FILE,
        )

    # ------------------------------------------------------------------ #
    # Archiving
    # ------------------------------------------------------------------ #

    def _archive_superseded_knowledge(self, cycle_dir: Path, archived_knowledge: str | None) -> None:
        """
        If the LLM flagged any content as outdated/superseded/redundant,
        append it to the permanent archive file, under a timestamped,
        cycle-referenced heading.

        This is intentionally APPEND-ONLY and never pruned itself: the
        active baseline is where token hygiene is actively enforced (via
        consolidation + the MAX_BASELINE_CHARS monitor below), but the
        archive is meant to be a permanent historical record - nothing that
        was once believed true should become unrecoverable just because it
        was superseded. A human (or a future audit tool) can always go back
        and see exactly what the lab used to believe, and when/why that
        changed.
        """
        if archived_knowledge is None or not archived_knowledge.strip():
            self.logger.debug("[KnowledgeCurator] No archived_knowledge for %s - nothing to archive.", cycle_dir.name)
            return

        config.KNOWLEDGE_BASE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = config.KNOWLEDGE_BASE_ARCHIVE_DIR / THEORY_ARCHIVE_FILENAME

        timestamp = datetime.now(timezone.utc).isoformat()
        entry_block = (
            f"\n## {cycle_dir.name} Archive\n"
            f"*Archived at {timestamp}*\n\n"
            f"{archived_knowledge.strip()}\n"
        )

        # Append rather than overwrite: theory_archive.md is a permanent,
        # ever-growing historical record - unlike theory_baseline.md, it is
        # NEVER consolidated or pruned itself. Open in append mode so every
        # prior cycle's archived content is preserved regardless of what was
        # already in the file.
        with archive_path.open("a", encoding="utf-8") as f:
            f.write(entry_block)

        self.logger.info(
            "[KnowledgeCurator] Archived superseded knowledge for %s to %s",
            cycle_dir.name,
            archive_path,
        )

    # ------------------------------------------------------------------ #
    # Token hygiene monitoring
    # ------------------------------------------------------------------ #

    def _check_token_hygiene(self, updated_theory_baseline: str) -> None:
        """
        Monitor (not enforce via truncation) the new baseline's size against
        MAX_BASELINE_CHARS. If it's still over budget after consolidation,
        log a loud, actionable warning - this is a signal for a human (or a
        future, more aggressive consolidation strategy) to intervene, not
        something this agent silently fixes by cutting content, which could
        destroy scientifically load-bearing information without any human
        awareness that it happened.
        """
        baseline_length = len(updated_theory_baseline)
        if baseline_length > MAX_BASELINE_CHARS:
            self.logger.warning(
                "[KnowledgeCurator] theory_baseline.md is %d characters, exceeding the "
                "MAX_BASELINE_CHARS budget of %d. Consider stronger consolidation/pruning "
                "on the next cycle to keep this file within local LLM context limits.",
                baseline_length,
                MAX_BASELINE_CHARS,
            )
        else:
            self.logger.debug(
                "[KnowledgeCurator] theory_baseline.md is %d/%d characters - within budget.",
                baseline_length,
                MAX_BASELINE_CHARS,
            )

    # ------------------------------------------------------------------ #
    # Shadow memory persistence
    # ------------------------------------------------------------------ #

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        system_prompt: str,
        prompt: str,
        previous_baseline_text: str,
        hypothesis_text: str,
        theory_update: TheoryUpdateModel,
    ) -> None:
        """
        Persist the full LLM exchange, the PREVIOUS baseline (for diffing
        purposes), the hypothesis that triggered this update, and the
        validated response - the complete audit trail for debugging how and
        why the baseline changed on this cycle, without needing to re-run
        the LLM.
        """
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "agent_name": self.agent_name,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "previous_baseline_text": previous_baseline_text,
            "hypothesis_text": hypothesis_text,
            "validated_response": theory_update.model_dump(),
            "new_baseline_char_count": len(theory_update.updated_theory_baseline),
            "exceeded_max_baseline_chars": len(theory_update.updated_theory_baseline) > MAX_BASELINE_CHARS,
        }

        shadow_path = shadow_dir / "knowledge_curator_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[KnowledgeCurator] Wrote shadow memory to %s", shadow_path)
