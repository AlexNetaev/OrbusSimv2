"""
hypothesis_architect.py
=========================
Team 4.2 - Hypothesis Architect Agent ("HypothesisArchitect").

Role in the pipeline:
-----------------------
This agent runs immediately after DataAnalyst and before Chef. Where
DataAnalyst answers "what happened, and was the hardware healthy?",
HypothesisArchitect answers the deeper scientific question: "WHY did it
happen, and what should we DO differently next time?"

It does this by combining two sources of context:

  1. `01_Knowledge_Base/theory_baseline.md` - the lab's current standing
     scientific understanding (whatever chemistry/physics theory it's
     operating from).
  2. `cycle_dir/C_Analysis/discrepancy.md` - this cycle's concrete outcome,
     as already assessed by DataAnalyst (healthy or not, what anomalies were
     found, what the kinetics looked like).

...and asking the LLM to bridge them: given what we believe about the
underlying science, what specific physical/chemical mechanism plausibly
explains this cycle's discrepancy, and what concrete parameter adjustment
should the next cycle try as a result? This hypothesis is exactly the kind
of input MachinePlanner needs (in a future refinement) to make each
successive cycle's plan smarter than a blind repeat of the last one.

Consistent with the rest of the system, this agent is stateless: everything
it needs comes from files already on disk, and its own output
(hypothesis.md + shadow memory) is written straight back to disk. Nothing is
retained on the instance across runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src import config
from src.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Filename DataAnalyst is expected to have written into cycle_dir/C_Analysis/.
DISCREPANCY_REPORT_FILENAME = "discrepancy.md"

# Filename this agent writes into cycle_dir/C_Analysis/ alongside discrepancy.md.
HYPOTHESIS_REPORT_FILENAME = "hypothesis.md"


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class HypothesisModel(BaseModel):
    """
    Strict output schema the LLM must populate: the scientific reasoning
    connecting theory to this cycle's observed outcome, plus a concrete,
    actionable recommendation.
    """

    root_cause_analysis: str = Field(
        ...,
        description=(
            "Physical/chemical explanation for the discrepancy between target and actual "
            "results, grounded in the theory baseline. If the cycle had no discrepancy (or "
            "no hardware data was available yet), state that explicitly rather than "
            "inventing a cause."
        ),
    )
    proposed_adjustment: str = Field(
        ...,
        description=(
            "Specific, concrete parameter adjustment(s) recommended for the next cycle "
            "(e.g. 'reduce target_temperature_c by 5C and increase mixing_time_s by 10s'). "
            "If no adjustment is warranted, say so explicitly rather than proposing a "
            "change for its own sake."
        ),
    )


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

class HypothesisArchitect(BaseAgent):
    """
    Reads the theory baseline and this cycle's discrepancy report; asks the
    LLM to form a grounded scientific hypothesis explaining the outcome and
    recommend a concrete adjustment; writes both a human-readable
    hypothesis.md and the full audit trail to shadow memory.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        # Fixed agent name per spec - "HypothesisArchitect" is this agent's
        # identity for logging/shadow-memory purposes system-wide.
        super().__init__(agent_name="HypothesisArchitect", workspace_path=workspace_path, model=model)

    # ------------------------------------------------------------------ #
    # BaseAgent contract
    # ------------------------------------------------------------------ #

    def run(self, cycle_dir: Path) -> None:
        """
        Single-pass entry point, invoked by `BaseAgent.execute()`.

        Flow:
            1. Read theory_baseline.md (global) and discrepancy.md (this
               cycle), each with a graceful placeholder if missing/empty.
            2. Ask the LLM for a HypothesisModel.
            3. Write hypothesis.md (human-readable) into cycle_dir/C_Analysis/.
            4. Write the full prompt/response audit trail to
               cycle_dir/D_Shadow_Memory/hypothesis_shadow.json.

        Design note - why this agent does NOT skip the LLM call when
        discrepancy.md is missing/degenerate:
            Unlike DataAnalyst (which genuinely has nothing to analyze
            without raw measurement data), a missing or "BLOCKED"/"TIMEOUT"
            discrepancy.md is still meaningful input for this agent: the
            correct hypothesis in that case is simply "no discrepancy to
            explain yet, no adjustment warranted" - which the LLM can state
            directly via HypothesisModel. Special-casing that as a Python-side
            early return would just be re-implementing, in code, a judgment
            the LLM can already make correctly from the prompt content itself.
        """
        analysis_dir = cycle_dir / "C_Analysis"
        shadow_memory_dir = cycle_dir / "D_Shadow_Memory"
        # Defensive: these should already exist (main_loop.create_cycle_directory
        # creates all four subdirs up front, and DataAnalyst runs before this
        # agent and also ensures C_Analysis exists), but we don't assume that
        # and create them idempotently ourselves too.
        analysis_dir.mkdir(parents=True, exist_ok=True)
        shadow_memory_dir.mkdir(parents=True, exist_ok=True)

        theory_text = self._read_text_or_placeholder(
            config.THEORY_BASELINE_FILE,
            placeholder="(theory_baseline.md is empty - no prior theoretical grounding available yet.)",
        )
        discrepancy_text = self._read_discrepancy_report(analysis_dir)

        prompt = self._build_hypothesis_prompt(theory_text=theory_text, discrepancy_text=discrepancy_text)
        system_prompt = (
            "You are the Hypothesis Architect for an autonomous self-driving laboratory. "
            "You do not run experiments or judge hardware health yourself - you take the "
            "lab's current theoretical understanding and this cycle's already-assessed "
            "outcome, and form a grounded scientific hypothesis connecting the two. Ground "
            "every claim in the theory baseline provided; do not invent chemistry or physics "
            "that isn't supported by it or by the observed data. If there is nothing to "
            "explain (no discrepancy, or no hardware data yet), say so plainly rather than "
            "manufacturing a root cause."
        )

        hypothesis = self.ask_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            response_model=HypothesisModel,
        )
        # `ask_llm` is typed to return `str | T`; when response_model is given
        # it always returns a validated T on success (or raises otherwise), so
        # this assertion is purely a static-typing aid, not a real runtime risk.
        assert isinstance(hypothesis, HypothesisModel)

        self._write_hypothesis_report(analysis_dir=analysis_dir, hypothesis=hypothesis)
        self._write_shadow_memory(
            shadow_memory_dir=shadow_memory_dir,
            system_prompt=system_prompt,
            prompt=prompt,
            theory_text=theory_text,
            discrepancy_text=discrepancy_text,
            hypothesis=hypothesis,
        )

        self.logger.info(
            "[HypothesisArchitect] Hypothesis complete | root_cause_analysis=%r",
            hypothesis.root_cause_analysis[:80] + ("..." if len(hypothesis.root_cause_analysis) > 80 else ""),
        )

    # ------------------------------------------------------------------ #
    # Reading inputs
    # ------------------------------------------------------------------ #

    def _read_text_or_placeholder(self, path: Path, placeholder: str) -> str:
        """
        Read a file's text content, or return a descriptive placeholder if
        the file doesn't exist or is empty. Mirrors the identical pattern
        used in `Chef` and `MachinePlanner` - all three agents need this for
        the exact same reason: an explicit placeholder is far clearer prompt
        context for the LLM than silent emptiness.
        """
        if not path.exists():
            self.logger.warning(
                "[HypothesisArchitect] Expected file missing: %s. Using placeholder text.", path
            )
            return placeholder

        text = path.read_text(encoding="utf-8").strip()
        return text if text else placeholder

    def _read_discrepancy_report(self, analysis_dir: Path) -> str:
        """
        Read `discrepancy.md` from this cycle's C_Analysis directory.

        If it's missing entirely (DataAnalyst hasn't run, crashed before
        writing, or was invoked against a differently-shaped directory), we
        log a warning and return an explicit placeholder rather than
        raising - per the design note in `run()`'s docstring, a missing
        discrepancy report is still valid, reasonable-to-explain input for
        this agent ("no discrepancy data is available yet"), not a reason to
        abort.
        """
        discrepancy_path = analysis_dir / DISCREPANCY_REPORT_FILENAME
        return self._read_text_or_placeholder(
            discrepancy_path,
            placeholder=(
                "(discrepancy.md not found - the Data Analyst may not have run yet, "
                "or hardware output was not available for this cycle.)"
            ),
        )

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _build_hypothesis_prompt(self, theory_text: str, discrepancy_text: str) -> str:
        """
        Assemble the full prompt: the lab's theory baseline + this cycle's
        discrepancy report, then ask for the structured hypothesis.
        """
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
   baseline above, plausibly explains the discrepancy between target and actual results
   this cycle? If there was no meaningful discrepancy (or no hardware data was available
   yet), state that explicitly rather than inventing an explanation.
2. `proposed_adjustment`: What SPECIFIC parameter adjustment(s) should the next cycle try
   as a result (name the actual parameter(s) and the direction/magnitude of change)? If no
   adjustment is warranted, say so explicitly.
"""

    # ------------------------------------------------------------------ #
    # Output persistence
    # ------------------------------------------------------------------ #

    def _write_hypothesis_report(self, analysis_dir: Path, hypothesis: HypothesisModel) -> None:
        """
        Write the human-readable Markdown hypothesis report. This sits
        alongside discrepancy.md in C_Analysis/, so anyone reviewing a
        cycle's analysis sees both "what happened" (discrepancy.md) and
        "why, and what to do next" (hypothesis.md) together.
        """
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
        shadow_memory_dir: Path,
        system_prompt: str,
        prompt: str,
        theory_text: str,
        discrepancy_text: str,
        hypothesis: HypothesisModel,
    ) -> None:
        """
        Persist the full LLM exchange and inputs as JSON, separate from the
        polished, human-facing hypothesis.md - the complete audit trail for
        debugging prompt quality or letting a future agent (e.g. a smarter
        MachinePlanner) re-examine exactly what was reasoned about without
        needing to re-run the LLM.
        """
        shadow_record = {
            "agent_name": self.agent_name,
            "model": self.model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "inputs": {
                "theory_baseline_text": theory_text,
                "discrepancy_text": discrepancy_text,
            },
            "validated_response": hypothesis.model_dump(),
        }

        shadow_path = shadow_memory_dir / "hypothesis_shadow.json"
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[HypothesisArchitect] Wrote shadow memory to %s", shadow_path)
