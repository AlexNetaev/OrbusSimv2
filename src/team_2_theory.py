"""
team_2_theory.py
==================
Team 2 - Theory & Curation ("The Long-Term Memory") agents.

Role in the pipeline:
-----------------------
Team 2 owns the lab's single most important piece of shared, cross-cycle
state: `01_Knowledge_Base/theory_baseline.md`, the document every future
MachinePlanner/DataAnalyst/HypothesisArchitect call reads in full. Three
agents run in strict sequence to maintain it responsibly:

  1. **Theoretician (Agent 2.1)** - the writer. It reads this cycle's fresh
     `hypothesis.md` alongside the CURRENT baseline and asks the LLM to
     produce a consolidated REPLACEMENT baseline that merges the new finding
     in - updating or replacing whatever it supersedes, not just appending
     on top. It overwrites `theory_baseline.md` directly.

  2. **FactChecker (Agent 2.2)** - the skeptic. It re-reads the baseline
     Theoretician JUST wrote and asks the LLM a single, focused question:
     does anything in here violate basic physics, or invent a constant that
     doesn't check out? If the answer is no, it raises `HallucinationException`
     - a hard stop, since a corrupted theory baseline poisons every future
     cycle that reads it until someone (or something) fixes it.

  3. **KnowledgeCurator (Agent 2.3)** - the token-hygiene manager. Purely
     deterministic on the common path: it checks the baseline's character
     length and, if it's under `MAX_BASELINE_CHARS`, does NOTHING and makes
     ZERO LLM calls. Only when the baseline has genuinely grown too large
     does it ask the LLM to compress it, moving the detailed, superseded
     material into the permanent `Archive/theory_archive.md` and leaving
     short reference pointers behind in the active baseline.

Why FactChecker is a separate agent from Theoretician, not a single call:
------------------------------------------------------------------------------
Theoretician's job is to synthesize - merge a new finding into existing
theory as coherently as possible. Asking the SAME call to also critically
doubt its own output invites exactly the kind of self-confirming reasoning
LLMs are prone to ("I wrote it, so it's probably right"). Splitting fact-
checking into its own, independently-invoked agent - with its own system
prompt oriented entirely around skepticism rather than synthesis - gives the
lab a genuine adversarial check rather than the same generative pass
grading its own homework.

Why KnowledgeCurator's length check is deterministic Python, not an LLM call:
------------------------------------------------------------------------------
Exactly the same reasoning already established by `SemanticSafetyAgent`
(src/team_5_execution.py) and DataAnalyst's reality check (src/team_4_
synthesis.py): a decision this codebase can make with total certainty in one
line of Python (`len(text) > MAX_BASELINE_CHARS`) should never be delegated
to an LLM call, which costs money/latency and introduces variance into an
otherwise perfectly deterministic gate. The LLM is only ever invoked for the
part that genuinely requires judgment - HOW to compress, not WHETHER to.

Naming note (see also `src/team_5_execution.py` / `src/team_4_synthesis.py`):
-----------------------------------------------------------------------------------
`src/agents/knowledge_curator.py` already defines a `KnowledgeCurator` class
that both consolidates AND archives in one pass. THIS module's `KnowledgeCurator`
has a narrower job - token hygiene ONLY, triggered by a length threshold - because
consolidation is now Theoretician's job instead. The two are intentionally
independent (no shared imports) and will need explicit aliasing wherever both
are wired into `main_loop.py` together, per the pattern already agreed for
Team 5's `MachinePlanner` and Team 4's `DataAnalyst`/`HypothesisArchitect`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src import config
from src.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

HYPOTHESIS_REPORT_FILENAME = "hypothesis.md"
FACT_CHECK_LOG_FILENAME = "fact_check_log.json"
THEORY_ARCHIVE_FILENAME = "theory_archive.md"

# Standard header written into theory_baseline.md the first time Theoretician
# ever encounters it empty/missing, so the file always has a consistent,
# parseable shape from cycle 1 onward.
STANDARD_BASELINE_HEADER = "# Theory Baseline\n\n*(No findings recorded yet.)*\n"

# Hard token-hygiene trigger for KnowledgeCurator: below this, it makes ZERO
# LLM calls and exits cleanly (see module docstring for why this check is
# deterministic Python, not an LLM judgment call).
MAX_BASELINE_CHARS = 4000


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class HallucinationException(Exception):
    """
    Raised by FactChecker when it judges the just-updated theory_baseline.md
    to contain a physics violation or a fabricated/implausible constant.

    A hard stop rather than an auto-correction: FactChecker's job is only to
    DETECT a problem, not to guess at a fix - inventing a "corrected" theory
    baseline on the spot would just be a second LLM call hallucinating on top
    of the first. Intended to be routed to `DeadlockManager` by the
    orchestrator (not implemented in this module).
    """

    def __init__(self, reasoning: str):
        super().__init__(f"FactChecker rejected theory_baseline.md: {reasoning}")
        self.reasoning = reasoning


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class TheoryBaselineUpdateModel(BaseModel):
    """
    Strict output schema for Theoretician: the complete, consolidated
    replacement for theory_baseline.md. Using a structured model here
    (rather than raw text, the way `src/team_3_simulation.py`'s
    PythonArchitect does for source code) gets this agent the same
    auto-correction retry benefit every other structured-output agent in
    this codebase gets, rather than relying on markdown-fence-stripping
    heuristics for a document this important.
    """

    updated_theory_baseline: str = Field(
        ...,
        description=(
            "The COMPLETE new version of theory_baseline.md, incorporating this cycle's "
            "hypothesis. This REPLACES the entire file - merge the new finding into the "
            "existing document rather than appending it; update or replace anything it "
            "supersedes; keep the result coherent and concise."
        ),
    )


class FactCheckResultModel(BaseModel):
    """Strict output schema for FactChecker's adversarial review."""

    is_valid: bool = Field(
        ...,
        description="False if the baseline contains a physics violation or a fabricated/implausible constant.",
    )
    reasoning: str = Field(..., description="Specific reasoning for the verdict - cite the exact claim if invalid.")


class CompressionResultModel(BaseModel):
    """
    Strict output schema for KnowledgeCurator's compression pass (only
    requested when the deterministic length check actually triggers it).
    """

    compressed_theory_baseline: str = Field(
        ...,
        description=(
            "The new, SHORTENED theory_baseline.md content: only the highly relevant current "
            "findings, plus short reference pointers to the archive for anything moved out."
        ),
    )
    archived_details: str = Field(
        ...,
        min_length=1,
        description=(
            "The detailed, older/superseded content being moved OUT of the active baseline and "
            "into the permanent archive. Must be non-empty - compression was triggered precisely "
            "because something needs to move out."
        ),
    )

    @field_validator("archived_details")
    @classmethod
    def _archived_details_not_blank(cls, value: str) -> str:
        """
        Reject whitespace-only content, not just the empty string -
        `min_length=1` alone would accept `" "`, which archives nothing of
        substance despite compression having been explicitly triggered.
        """
        if not value.strip():
            raise ValueError("archived_details must not be empty or whitespace-only when compression is triggered.")
        return value


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
        logger.warning("[team_2_theory] Expected file missing: %s. Using placeholder text.", path)
        return placeholder

    text = path.read_text(encoding="utf-8").strip()
    return text if text else placeholder


# --------------------------------------------------------------------------- #
# Agent 2.1: Theoretician
# --------------------------------------------------------------------------- #

class Theoretician(BaseAgent):
    """
    Reads this cycle's hypothesis.md and the current theory_baseline.md;
    asks the LLM to produce a consolidated replacement baseline; overwrites
    theory_baseline.md; writes the full audit trail to shadow memory.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(agent_name="Theoretician", workspace_path=workspace_path, model=model)

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read (and, if missing/empty, initialize on disk) the current
               theory_baseline.md, and this cycle's hypothesis.md, each with
               a graceful placeholder if unavailable.
            2. Ask the LLM for a TheoryBaselineUpdateModel: a full
               consolidated replacement.
            3. Overwrite theory_baseline.md.
            4. Write the full prompt/response audit trail to shadow memory.
        """
        current_baseline_text = self._read_or_initialize_baseline()
        hypothesis_text = _read_text_or_placeholder(
            cycle_dir / "C_Analysis" / HYPOTHESIS_REPORT_FILENAME,
            placeholder=(
                "(hypothesis.md not found for this cycle - the Hypothesis Architect may not "
                "have run yet, or produced no output. No new finding to incorporate.)"
            ),
        )

        prompt = self._build_consolidation_prompt(
            current_baseline_text=current_baseline_text, hypothesis_text=hypothesis_text
        )
        system_prompt = (
            "You are the Theoretician for an autonomous self-driving laboratory. You maintain "
            "the lab's single, permanent theory baseline document - the consolidated scientific "
            "understanding every future cycle's planning and analysis relies on. When "
            "incorporating a new finding, MERGE it into the existing document: update or replace "
            "any statement it supersedes, remove redundancy, and keep the result coherent and "
            "concise. Never simply append the new finding on top of the old text. If nothing in "
            "the current baseline is actually outdated by this cycle's finding, leave it in "
            "place and only add what's genuinely new."
        )

        theory_update = self.ask_llm(
            prompt=prompt, system_prompt=system_prompt, response_model=TheoryBaselineUpdateModel
        )
        assert isinstance(theory_update, TheoryBaselineUpdateModel)

        self._write_updated_baseline(theory_update.updated_theory_baseline)
        self._write_shadow_memory(
            cycle_dir=cycle_dir,
            system_prompt=system_prompt,
            prompt=prompt,
            previous_baseline_text=current_baseline_text,
            hypothesis_text=hypothesis_text,
            theory_update=theory_update,
        )

        self.logger.info(
            "[Theoretician] Baseline updated for %s | new_length=%d chars",
            cycle_dir.name,
            len(theory_update.updated_theory_baseline),
        )

    def _read_or_initialize_baseline(self) -> str:
        """
        Read theory_baseline.md, writing STANDARD_BASELINE_HEADER to disk
        FIRST if it's missing/empty. Theoretician owns this file's
        lifecycle, so it should never leave it in a genuinely uninitialized
        state once it has run.
        """
        if not config.THEORY_BASELINE_FILE.exists():
            self.logger.warning(
                "[Theoretician] theory_baseline.md not found at %s - creating with standard header.",
                config.THEORY_BASELINE_FILE,
            )
            config.THEORY_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
            config.THEORY_BASELINE_FILE.write_text(STANDARD_BASELINE_HEADER, encoding="utf-8")
            return STANDARD_BASELINE_HEADER

        text = config.THEORY_BASELINE_FILE.read_text(encoding="utf-8")
        if not text.strip():
            self.logger.warning(
                "[Theoretician] theory_baseline.md at %s is empty - initializing with standard header.",
                config.THEORY_BASELINE_FILE,
            )
            config.THEORY_BASELINE_FILE.write_text(STANDARD_BASELINE_HEADER, encoding="utf-8")
            return STANDARD_BASELINE_HEADER

        return text

    def _build_consolidation_prompt(self, current_baseline_text: str, hypothesis_text: str) -> str:
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
Produce `updated_theory_baseline`: the COMPLETE, new version of theory_baseline.md. Merge
this cycle's finding into the existing document rather than appending it - update or replace
any existing statement the new finding supersedes, remove redundancy, and keep the result
coherent, well-organized, and as concise as possible while losing no scientifically
load-bearing content. This is a full replacement document, not a diff.
"""

    def _write_updated_baseline(self, updated_theory_baseline: str) -> None:
        config.THEORY_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.THEORY_BASELINE_FILE.write_text(updated_theory_baseline, encoding="utf-8")
        self.logger.info(
            "[Theoretician] Wrote updated theory baseline (%d chars) to %s",
            len(updated_theory_baseline),
            config.THEORY_BASELINE_FILE,
        )

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        system_prompt: str,
        prompt: str,
        previous_baseline_text: str,
        hypothesis_text: str,
        theory_update: TheoryBaselineUpdateModel,
    ) -> None:
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
        }

        shadow_path = shadow_dir / "team2_theoretician_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[Theoretician] Wrote shadow memory to %s", shadow_path)


# --------------------------------------------------------------------------- #
# Agent 2.2: FactChecker
# --------------------------------------------------------------------------- #

class FactChecker(BaseAgent):
    """
    Re-reads the just-updated theory_baseline.md and asks the LLM a single,
    adversarially-framed question: does anything here violate basic physics
    or invent an implausible constant? Raises `HallucinationException` on a
    negative verdict. Always appends a lean entry to the global,
    `01_Knowledge_Base/fact_check_log.json` running log, and writes the full
    per-cycle exchange to shadow memory.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(agent_name="FactChecker", workspace_path=workspace_path, model=model)

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read theory_baseline.md fresh from disk (whatever Theoretician
               most recently wrote - this agent never trusts an in-memory
               handoff, consistent with the whole system's stateless design).
            2. Ask the LLM for a FactCheckResultModel.
            3. Append a lean entry to the global fact_check_log.json
               (ALWAYS, regardless of verdict).
            4. Write the full exchange to per-cycle shadow memory (ALWAYS).
            5. If is_valid is False, raise HallucinationException AFTER both
               of the above have already been persisted.
        """
        baseline_text = _read_text_or_placeholder(
            config.THEORY_BASELINE_FILE,
            placeholder="(theory_baseline.md is empty - nothing to fact-check yet.)",
        )

        prompt = self._build_fact_check_prompt(baseline_text=baseline_text)
        system_prompt = (
            "You are the Fact Checker for an autonomous self-driving laboratory - a deliberately "
            "skeptical reviewer, not the author of this document. Your only job is to look for "
            "violations of basic physical/chemical law, or numeric constants that are implausible "
            "or fabricated (e.g. a reaction rate constant with an impossible order of magnitude, a "
            "temperature below absolute zero, a stated efficiency above 100%). Do not rewrite or "
            "improve the document - only judge it. Give the document the benefit of the doubt on "
            "genuinely uncertain or qualitative claims; flag only concrete, checkable violations."
        )

        result = self.ask_llm(prompt=prompt, system_prompt=system_prompt, response_model=FactCheckResultModel)
        assert isinstance(result, FactCheckResultModel)

        self._append_to_fact_check_log(cycle_dir=cycle_dir, baseline_text=baseline_text, result=result)
        self._write_shadow_memory(
            cycle_dir=cycle_dir, system_prompt=system_prompt, prompt=prompt, baseline_text=baseline_text, result=result
        )

        self.logger.info("[FactChecker] Verdict for %s: is_valid=%s", cycle_dir.name, result.is_valid)

        if not result.is_valid:
            raise HallucinationException(reasoning=result.reasoning)

    def _build_fact_check_prompt(self, baseline_text: str) -> str:
        return f"""Review this theory baseline for physics violations or fabricated constants.

## Theory Baseline (theory_baseline.md, as just updated)
```
{baseline_text}
```

## Your task
Determine:
1. `is_valid`: True unless you find a concrete violation of basic physical/chemical law, or a
   numeric constant that is implausible or fabricated. False if you find such a violation.
2. `reasoning`: Your specific reasoning for the verdict. If invalid, cite the exact claim and
   explain precisely what is wrong with it.
"""

    def _append_to_fact_check_log(self, cycle_dir: Path, baseline_text: str, result: FactCheckResultModel) -> None:
        """
        Append a lean entry to the GLOBAL, ever-growing fact_check_log.json
        (config.KNOWLEDGE_BASE_DIR, not cycle-scoped - per spec). Kept
        deliberately lean (verdict + reasoning + a character-count marker,
        NOT the full baseline text) so this log doesn't become its own
        unbounded-growth problem parallel to the one KnowledgeCurator (2.3)
        exists to solve for theory_baseline.md itself - nothing in this
        codebase reads this log back into a prompt, but there's no reason to
        let it grow without bound regardless. The full baseline text that
        was actually checked IS preserved, per-cycle, in shadow memory below.
        """
        log_path = config.KNOWLEDGE_BASE_DIR / FACT_CHECK_LOG_FILENAME
        log_path.parent.mkdir(parents=True, exist_ok=True)

        entries: list[dict] = []
        if log_path.exists():
            existing_text = log_path.read_text(encoding="utf-8").strip()
            if existing_text:
                try:
                    entries = json.loads(existing_text)
                    if not isinstance(entries, list):
                        # Defensive: if a prior write ever produced something
                        # other than a JSON array, don't crash - start a
                        # fresh log rather than losing this cycle's entry.
                        self.logger.warning(
                            "[FactChecker] %s did not contain a JSON array - starting a fresh log.", log_path
                        )
                        entries = []
                except json.JSONDecodeError:
                    self.logger.warning("[FactChecker] %s is not valid JSON - starting a fresh log.", log_path)
                    entries = []

        entries.append(
            {
                "cycle_id": cycle_dir.name,
                "is_valid": result.is_valid,
                "reasoning": result.reasoning,
                "theory_baseline_chars_checked": len(baseline_text),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        self.logger.info("[FactChecker] Appended entry to %s (%d total entries)", log_path, len(entries))

    def _write_shadow_memory(
        self, cycle_dir: Path, system_prompt: str, prompt: str, baseline_text: str, result: FactCheckResultModel
    ) -> None:
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "agent_name": self.agent_name,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "theory_baseline_text_checked": baseline_text,
            "validated_response": result.model_dump(),
        }

        shadow_path = shadow_dir / "team2_fact_checker_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[FactChecker] Wrote shadow memory to %s", shadow_path)


# --------------------------------------------------------------------------- #
# Agent 2.3: KnowledgeCurator
# --------------------------------------------------------------------------- #

class KnowledgeCurator(BaseAgent):
    """
    Token-hygiene manager. Checks theory_baseline.md's length with plain
    Python; if it's under MAX_BASELINE_CHARS, does nothing and makes ZERO
    LLM calls. Otherwise, asks the LLM to compress it, archives the
    superseded detail, and rewrites the baseline to just the compressed
    result plus reference pointers.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        super().__init__(agent_name="KnowledgeCurator", workspace_path=workspace_path, model=model)

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read theory_baseline.md and check its length (plain Python).
            2. If under MAX_BASELINE_CHARS: write a lightweight shadow
               memory record noting no compression was needed, and return.
               ZERO LLM calls in this branch.
            3. If over: ask the LLM for a CompressionResultModel, append
               archived_details to Archive/theory_archive.md (with a
               timestamp and cycle reference), rewrite theory_baseline.md
               with compressed_theory_baseline, and write full shadow memory.
        """
        if not config.THEORY_BASELINE_FILE.exists():
            self.logger.warning(
                "[KnowledgeCurator] theory_baseline.md not found at %s - nothing to curate.",
                config.THEORY_BASELINE_FILE,
            )
            self._write_shadow_memory(cycle_dir=cycle_dir, action="no_op_missing_baseline", baseline_length=0)
            return

        baseline_text = config.THEORY_BASELINE_FILE.read_text(encoding="utf-8")
        baseline_length = len(baseline_text)

        if baseline_length <= MAX_BASELINE_CHARS:
            self.logger.info(
                "[KnowledgeCurator] theory_baseline.md is %d/%d characters - within budget, no compression needed.",
                baseline_length,
                MAX_BASELINE_CHARS,
            )
            self._write_shadow_memory(
                cycle_dir=cycle_dir, action="no_compression_needed", baseline_length=baseline_length
            )
            return

        self.logger.warning(
            "[KnowledgeCurator] theory_baseline.md is %d characters, exceeding MAX_BASELINE_CHARS=%d - "
            "requesting compression.",
            baseline_length,
            MAX_BASELINE_CHARS,
        )

        prompt = self._build_compression_prompt(baseline_text=baseline_text)
        system_prompt = (
            "You are the Knowledge Curator for an autonomous self-driving laboratory, responsible "
            "purely for TOKEN HYGIENE - not scientific judgment. The active theory baseline has "
            "grown too large. Identify the older, more detailed, or superseded material that can "
            "be moved into a permanent archive, leaving only the highly relevant CURRENT findings "
            "plus short reference pointers ('see archive for full derivation of X') in the active "
            "document. Never discard information - only relocate it out of the active document."
        )

        compression_result = self.ask_llm(
            prompt=prompt, system_prompt=system_prompt, response_model=CompressionResultModel
        )
        assert isinstance(compression_result, CompressionResultModel)

        self._archive_details(cycle_dir=cycle_dir, archived_details=compression_result.archived_details)
        self._write_compressed_baseline(compression_result.compressed_theory_baseline)

        new_length = len(compression_result.compressed_theory_baseline)
        if new_length > MAX_BASELINE_CHARS:
            # Monitoring only, never enforced by further truncation - same
            # fail-safe philosophy as every other token-hygiene check in
            # this codebase: warn loudly, never silently destroy content.
            self.logger.warning(
                "[KnowledgeCurator] theory_baseline.md is STILL %d characters after compression "
                "(budget %d) - consider stronger compression on the next trigger.",
                new_length,
                MAX_BASELINE_CHARS,
            )

        self._write_shadow_memory(
            cycle_dir=cycle_dir,
            action="compressed",
            baseline_length=baseline_length,
            system_prompt=system_prompt,
            prompt=prompt,
            compression_result=compression_result,
            new_baseline_length=new_length,
        )

    def _build_compression_prompt(self, baseline_text: str) -> str:
        return f"""The active theory baseline has exceeded its size budget. Compress it.

## Current Theory Baseline (theory_baseline.md, {len(baseline_text)} characters)
```
{baseline_text}
```

## Your task
Produce:
1. `compressed_theory_baseline`: A SHORTENED version containing only the highly relevant
   CURRENT findings, plus short reference pointers to the archive for anything moved out
   (e.g. "see Archive/theory_archive.md for the full derivation of the Arrhenius fit").
2. `archived_details`: The detailed, older, or superseded material being moved OUT of the
   active document. This must capture everything of substance that `compressed_theory_baseline`
   no longer contains in full - nothing should be silently lost, only relocated.
"""

    def _archive_details(self, cycle_dir: Path, archived_details: str) -> None:
        config.KNOWLEDGE_BASE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = config.KNOWLEDGE_BASE_ARCHIVE_DIR / THEORY_ARCHIVE_FILENAME

        timestamp = datetime.now(timezone.utc).isoformat()
        entry_block = (
            f"\n## {cycle_dir.name} Archive (Token Hygiene Compression)\n"
            f"*Archived at {timestamp}*\n\n{archived_details.strip()}\n"
        )

        with archive_path.open("a", encoding="utf-8") as f:
            f.write(entry_block)

        self.logger.info(
            "[KnowledgeCurator] Archived compressed-out detail for %s to %s", cycle_dir.name, archive_path
        )

    def _write_compressed_baseline(self, compressed_theory_baseline: str) -> None:
        config.THEORY_BASELINE_FILE.write_text(compressed_theory_baseline, encoding="utf-8")
        self.logger.info(
            "[KnowledgeCurator] Wrote compressed theory baseline (%d chars) to %s",
            len(compressed_theory_baseline),
            config.THEORY_BASELINE_FILE,
        )

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        action: str,
        baseline_length: int,
        system_prompt: str | None = None,
        prompt: str | None = None,
        compression_result: CompressionResultModel | None = None,
        new_baseline_length: int | None = None,
    ) -> None:
        """
        Persist the audit trail for every outcome, including the "did
        nothing" case - so it's always visible in the record whether
        compression was needed/triggered for a given cycle, not just when it
        actually happened.
        """
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "agent_name": self.agent_name,
            "model": self.model,
            "action": action,
            "max_baseline_chars": MAX_BASELINE_CHARS,
            "baseline_length_before": baseline_length,
            "llm_called": compression_result is not None,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "validated_response": compression_result.model_dump() if compression_result is not None else None,
            "baseline_length_after": new_baseline_length,
        }

        shadow_path = shadow_dir / "team2_knowledge_curator_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[KnowledgeCurator] Wrote shadow memory to %s", shadow_path)
