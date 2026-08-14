"""
deadlock_manager.py
=====================
Step 8 (original) / Step 5 (Grand Orchestration) - Deadlock Management &
Chain-of-Thought (CoT) Voting System.

Role in the system:
---------------------
This module is the Self-Driving Lab's autonomous circuit breaker. Unlike a
conventional application, this lab is meant to run unattended for long
stretches - there is no human sitting at a GUI ready to click "retry" or
"abort" when something goes wrong mid-cycle. When an unhandled error occurs
(a hardware timeout, an LLM that repeatedly fails schema validation, a
persistent Gatekeeper VETO, a rejected theory baseline, or any other
unexpected runtime failure), crashing the whole process is the worst possible
outcome, and blocking on a human prompt is not an option at all.

Instead, `DeadlockManager` gives the lab a structured way to decide for
itself what to do next:

  1. **Crisis Proposal Generation** - `ChefAgent` (the lab's coordinator,
     `src/agents/chef.py`) is asked to look at what went wrong and propose
     exactly three concrete, actionable recovery paths.
  2. **Chain-of-Thought Voting** - three LEAD agents from across the full
     blueprint are each independently asked to reason about the crisis and
     the three options from their own domain perspective, then cast a vote
     with their reasoning attached:
       - `PythonArchitect` (Team 3.1 - simulation)
       - `HypothesisArchitect` (Team 4.2 - scientific synthesis)
       - `MachinePlanner` (Team 5.1 - hardware compilation)
  3. **Tally & Resolution** - the votes are tallied (simple majority wins);
     a full, human-readable protocol and machine-readable audit trail are
     written to disk; and the winning option is returned to the caller
     (`main_loop.py`) to act on.

This is NOT a mechanism for the agents to "negotiate" or exchange messages
with each other - consistent with the rest of this system, every agent call
here is a single, independent, stateless LLM invocation. The only
"coordination" happening is DeadlockManager itself collecting and tallying
independently-cast votes, exactly the way a human committee vote works: each
member reasons privately and votes; only the tally is shared.

Import aliasing note:
------------------------
`PythonArchitect`, `HypothesisArchitect`, and `MachinePlanner` are imported
here from their Team 3/4/5 blueprint modules specifically (NOT from the
legacy `src/agents/` package, which defines classes of some of the same
names playing different, incompatible roles). `ChefAgent` is imported from
`src/agents/chef.py` - the legacy Chef already fulfills the blueprint's
"ChefAgent (The Coordinator)" role exactly, so it is reused rather than
reimplemented, per this project's ongoing plan to keep BaseAgent,
llm_wrapper, DeadlockManager, and (per the same logic) Chef as shared,
already-robust foundation across both the legacy and full-blueprint
architectures.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src import config
from src.agents.chef import Chef as ChefAgent
from src.team_3_simulation import PythonArchitect
from src.team_4_synthesis import HypothesisArchitect as ScientificHypothesisArchitect
from src.team_5_execution import MachinePlanner as CompilerMachinePlanner

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Reserved exception types
# --------------------------------------------------------------------------- #

class HardwareTimeoutException(Exception):
    """
    Reserved for a hardware-execution layer that treats a hardware timeout
    as a hard failure. Not currently raised anywhere in this codebase (the
    legacy DataAnalyst's polling logs a warning and writes a timeout notice
    rather than raising; the blueprint's `wait_for_hardware` mock in
    `main_loop.py` never times out at all). Kept defined so
    `main_loop.py`'s exception handling can name it explicitly and route it
    through `DeadlockManager` the same way as every other unhandled cycle
    failure, whenever something in the pipeline actually raises it.
    """
    pass


@dataclass(frozen=True)
class DeadlockResolution:
    """Maschinenlesbares Ergebnis der Deadlock-Auflösung."""
    option_id: str
    description: str
    action: Literal["continue_next_cycle", "terminate_project", "simulation_only_cycle", "retry_with_adjustment"]

    def __str__(self) -> str:
        return f"{self.option_id}: {self.description}"

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Fixed, canonical set of option identifiers. Every crisis always produces
# exactly these three IDs, in this order - this lets VoteModel constrain
# `vote` to a static Literal (validated automatically by Pydantic on every
# vote) without needing any dynamic, per-crisis schema injection into the
# LLM call.
OPTION_IDS: tuple[str, ...] = ("Option_A", "Option_B", "Option_C")

# The three LEAD agents queried for votes, per the full blueprint, in the
# fixed order they're always polled. ChefAgent deliberately does NOT get an
# ordinary vote here - it already authored the options (Step A) and is
# reserved as the TIE-BREAKER of last resort (see _tally_votes), so that
# Chef's influence on the outcome is exactly one vote's worth in aggregate,
# not two.
VOTING_AGENT_CLASSES: tuple[type, ...] = (PythonArchitect, ScientificHypothesisArchitect, CompilerMachinePlanner)

# Filenames this module writes.
CYCLE_PROTOCOL_FILENAME = "cycle_protocol.md"
DEADLOCK_SHADOW_FILENAME = "deadlock_shadow.json"


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class OptionModel(BaseModel):
    """One proposed recovery path."""

    id: Literal["Option_A", "Option_B", "Option_C"] = Field(
        ..., description="Fixed identifier for this option - must be exactly Option_A, Option_B, or Option_C."
    )
    description: str = Field(..., description="Concrete, actionable description of this recovery path.")


class CrisisOptionsModel(BaseModel):
    """
    Strict output schema for crisis proposal generation: exactly three
    recovery options, identified Option_A/B/C in that fixed order.
    """

    options: list[OptionModel] = Field(..., description="Exactly 3 proposed recovery options.")

    @field_validator("options")
    @classmethod
    def _exactly_three_in_fixed_order(cls, options: list[OptionModel]) -> list[OptionModel]:
        """
        Reject (rather than silently reshape) any response that doesn't
        contain exactly 3 options in the fixed Option_A/B/C order. Raising
        here surfaces a ValidationError that llm_wrapper.py's
        auto-correction retry loop will catch and feed back to the model as
        a concrete error to fix.
        """
        if len(options) != 3:
            raise ValueError(f"options must contain exactly 3 entries, got {len(options)}.")
        actual_ids = tuple(option.id for option in options)
        if actual_ids != OPTION_IDS:
            raise ValueError(f"options must be exactly {OPTION_IDS} in that order, got {actual_ids}.")
        return options


class VoteModel(BaseModel):
    """
    Strict output schema for a single agent's vote, per the blueprint's
    structure: chain-of-thought analysis FIRST, then the strict JSON vote,
    e.g. `{"analysis": "...", "vote": "Option_B"}`.

    Field ORDER matters here beyond just readability: because Pydantic
    models are typically serialized (and, more importantly, typically
    generated by an LLM) in field-declaration order, putting `analysis`
    before `vote` encourages the underlying model to reason its way to a
    conclusion before committing to one - the same "reasoning before the
    answer" principle used throughout this codebase's other CoT-style
    models (e.g. `ChefDecisionModel`).

    `vote` uses the same static `Literal["Option_A", "Option_B", "Option_C"]`
    constraint as `OptionModel.id` - any other value fails Pydantic
    validation immediately (triggering llm_wrapper's auto-correction retry
    loop), which is what "must strictly match one of the option IDs" means
    in practice here.
    """

    analysis: str = Field(
        ..., description="Chain-of-thought analysis explaining why this option was chosen over the others."
    )
    vote: Literal["Option_A", "Option_B", "Option_C"] = Field(
        ..., description="The single option ID this agent votes for."
    )


# --------------------------------------------------------------------------- #
# The manager
# --------------------------------------------------------------------------- #

class DeadlockManager:
    """
    Autonomous circuit breaker: generates recovery options for a crisis,
    collects independent chain-of-thought votes from three lead agents
    drawn from across the full blueprint, tallies the result, and persists
    a full human- and machine-readable record of the whole process.

    Not a BaseAgent subclass - DeadlockManager doesn't perform a single
    cycle's worth of scientific work the way the lead agents it queries do.
    It's an orchestration utility that CONSTRUCTS and QUERIES those other
    agents directly (via their inherited `ask_llm()` helper), the same way
    `main_loop.py` constructs and executes them - just for a different
    purpose (voting) than their normal `run()` entry point.
    """

    def __init__(self, workspace_path: Path | None = None, model: str = config.OLLAMA_MODEL) -> None:
        self.workspace_path: Path = (workspace_path or config.WORKSPACE_ROOT).resolve()
        self.model: str = model
        self.logger = logging.getLogger("deadlock_manager")

    # ------------------------------------------------------------------ #
    # Step A: Crisis Proposal Generation
    # ------------------------------------------------------------------ #

    def generate_crisis_options(self, reason: str, cycle_dir: Path) -> CrisisOptionsModel:
        """
        Ask ChefAgent to analyze the crisis and propose exactly 3 actionable
        recovery paths.

        Args:
            reason: Human-readable description of what went wrong - for the
                    full blueprint, `main_loop.py` builds an exception-
                    type-specific string here (e.g. citing the exact
                    hardware safety violations, or the simulation's captured
                    stderr) rather than a generic `str(exception)`, so
                    ChefAgent and the voting agents get real context.
            cycle_dir: The cycle directory the crisis occurred in - used only
                       to gather best-effort context (what files already
                       exist for this cycle), not written to by this method.

        Returns:
            A validated CrisisOptionsModel with exactly 3 options.
        """
        context_block = self._gather_crisis_context(cycle_dir)

        prompt = f"""A crisis has occurred during an autonomous self-driving laboratory cycle. Analyze it and propose a recovery plan.

## Crisis Reason
{reason}

## Context From This Cycle
{context_block}

## Your task
Propose EXACTLY 3 distinct, concrete, actionable recovery paths for this crisis, labeled
Option_A, Option_B, and Option_C in that order. Each option's description must be specific
enough that another agent could act on it directly (e.g. "retry this cycle's hardware run
with target_temperature_c reduced by 10C" rather than a vague "try again"). The three options should represent genuinely different strategies (e.g. retry as-is, retry with an
adjustment, or abandon this cycle and proceed to the next one) rather than minor variations of the same idea.
"""
        system_prompt = (
            "You are ChefAgent, acting in your role as crisis coordinator for an autonomous "
            "self-driving laboratory. When something goes wrong and the lab cannot proceed "
            "normally, you are responsible for proposing a small, concrete set of recovery "
            "options for the rest of the lab to vote on. Be decisive and specific - vague or "
            "overlapping options are not useful to a voting committee."
        )

        chef = ChefAgent(workspace_path=self.workspace_path, model=self.model)
        crisis_options = chef.ask_llm(prompt=prompt, system_prompt=system_prompt, response_model=CrisisOptionsModel)
        # `ask_llm` is typed to return `str | T`; when response_model is given
        # it always returns a validated T on success (or raises otherwise), so
        # this assertion is purely a static-typing aid, not a real runtime risk.
        assert isinstance(crisis_options, CrisisOptionsModel)

        self.logger.info(
            "[DeadlockManager] Generated %d crisis options for %s.", len(crisis_options.options), cycle_dir.name
        )
        return crisis_options

    def _gather_crisis_context(self, cycle_dir: Path) -> str:
        """
        Best-effort, lightweight context for the crisis prompt: which
        analysis/shadow-memory/simulation/hardware files already exist for
        this cycle (names only, not full contents - keeping this prompt
        small and fast) plus the global directive, so ChefAgent's proposed
        options stay aligned with the lab's actual goal rather than being
        generic.

        Deliberately tolerant of a partially-built or even nonexistent
        cycle_dir - this never raises, it just reports what it can find.
        """
        directive_text = "(directive.md unavailable)"
        if config.DIRECTIVE_FILE.exists():
            text = config.DIRECTIVE_FILE.read_text(encoding="utf-8").strip()
            directive_text = text if text else "(directive.md is empty)"

        existing_files: list[str] = []
        for subdirectory_name in ("A_Simulation", "B_Hardware", "C_Analysis", "D_Shadow_Memory"):
            subdirectory = cycle_dir / subdirectory_name
            if subdirectory.exists():
                existing_files.extend(
                    f"{subdirectory_name}/{f.name}" for f in sorted(subdirectory.iterdir()) if f.is_file()
                )

        files_block = (
            "\n".join(f"- {name}" for name in existing_files) if existing_files else "(no files yet for this cycle)"
        )

        return f"Global Directive: {directive_text}\n\nFiles already present for {cycle_dir.name}:\n{files_block}"

    # ------------------------------------------------------------------ #
    # Step B: Chain-of-Thought Voting
    # ------------------------------------------------------------------ #

    def collect_agent_votes(
        self,
        options: CrisisOptionsModel,
        cycle_dir: Path,
        reason: str,
    ) -> dict[str, VoteModel]:
        """
        Query the three blueprint LEAD agents independently for a
        chain-of-thought vote on the proposed crisis options:
        `PythonArchitect` (Team 3.1), `HypothesisArchitect` (Team 4.2), and
        `MachinePlanner` (Team 5.1).

        Note on signature: this includes `reason` in addition to `options`
        and `cycle_dir`. A vote is meaningless without knowing what crisis is
        being voted on, so `reason` is required here rather than expecting
        each voting call to somehow rediscover it from disk.

        Each agent is queried using ITS OWN class (via a fresh, stateless
        instance's inherited `ask_llm()`), with a role-specific system prompt
        framing the vote from that agent's own domain perspective in the
        full blueprint - PythonArchitect thinks about whether the
        simulation alone can address the crisis, HypothesisArchitect thinks
        about scientific value, and MachinePlanner thinks about hardware
        safety/feasibility. This mirrors how a real committee vote works:
        each member brings their own expertise to bear on the same shared
        question, independently.

        Args:
            options: The CrisisOptionsModel produced by generate_crisis_options.
            cycle_dir: The cycle directory the crisis occurred in.
            reason: The crisis reason, passed through to every voter's prompt.

        Returns:
            {agent_name: VoteModel}, one entry per voting agent, in the fixed
            order defined by VOTING_AGENT_CLASSES.
        """
        options_block = self._format_options_block(options)
        votes: dict[str, VoteModel] = {}

        for agent_class in VOTING_AGENT_CLASSES:
            agent = agent_class(workspace_path=self.workspace_path, model=self.model)
            role_framing = self._role_framing_for(agent.agent_name)

            prompt = f"""A crisis has occurred during this research cycle. Vote on how to recover.

## Crisis Reason
{reason}

## Proposed Recovery Options
{options_block}

## Your task
{role_framing}
Reason step-by-step about the trade-offs of each option from your own perspective (this is
your `analysis`), then select EXACTLY ONE option by its ID as your `vote` (Option_A,
Option_B, or Option_C).
"""
            system_prompt = (
                f"You are casting one independent vote in a recovery-decision committee for an "
                f"autonomous self-driving laboratory, in your role as {agent.agent_name}. Reason "
                f"carefully and specifically before voting - your analysis will be recorded "
                f"alongside your vote as part of the lab's permanent audit trail."
            )

            vote = agent.ask_llm(prompt=prompt, system_prompt=system_prompt, response_model=VoteModel)
            # See note in generate_crisis_options() re: this assertion's purpose.
            assert isinstance(vote, VoteModel)

            votes[agent.agent_name] = vote
            self.logger.info("[DeadlockManager] %s voted %s.", agent.agent_name, vote.vote)

        return votes

    def _format_options_block(self, options: CrisisOptionsModel) -> str:
        """Render the crisis options as a labeled block for prompts."""
        return "\n".join(f"- {option.id}: {option.description}" for option in options.options)

    def _role_framing_for(self, agent_name: str) -> str:
        """
        Return a short, role-specific instruction framing how this agent
        should approach the vote, reflecting its normal domain
        responsibility in the FULL BLUEPRINT pipeline. Falls back to a
        generic framing for any agent name not explicitly covered
        (defensive - VOTING_AGENT_CLASSES is the single source of truth for
        who actually votes, but this keeps the helper safe to call with an
        arbitrary name too).
        """
        role_framings = {
            "PythonArchitect": (
                "As the Python Architect, weigh each option primarily in terms of whether the "
                "digital-twin simulation itself can be adjusted or re-run to address this crisis, "
                "without requiring new physical hardware resources - which option best preserves "
                "that possibility?"
            ),
            "HypothesisArchitect": (
                "As the Hypothesis Architect, weigh each option primarily in terms of scientific "
                "value - which option best preserves the chance of learning something useful this "
                "cycle, or avoids drawing a false conclusion from a compromised run?"
            ),
            "MachinePlanner": (
                "As the Machine Planner, weigh each option primarily in terms of hardware safety "
                "and physical feasibility - which option best avoids further equipment risk or "
                "invalid parameter states?"
            ),
        }
        return role_framings.get(
            agent_name,
            "Weigh each option on its overall merit for recovering this cycle successfully.",
        )

    # ------------------------------------------------------------------ #
    # Step C: Tally & Resolution
    # ------------------------------------------------------------------ #

    def resolve_deadlock(self, reason: str, cycle_dir: Path) -> str:
        """
        Full end-to-end deadlock resolution: generate options, collect
        votes, tally the result, persist the protocol and audit trail, and
        return the winning resolution.

        This method is designed for complete fallback safety: every step
        that could plausibly raise (crisis generation, each vote, the
        tie-break vote) is allowed to raise NORMALLY up through this method
        for everything except the tie-break vote specifically - the
        tie-break vote is wrapped so that if IT fails for any reason, we
        fall back to the deterministic `Option_A` default rather than
        letting the crisis-resolution process itself become a second
        source of unhandled failure.

        Args:
            reason: Human-readable description of what went wrong.
            cycle_dir: The cycle directory the crisis occurred in.

        Returns:
            A human-readable string naming the winning option and its
            description, e.g. "Option_B: Retry this cycle's hardware run
            with target_temperature_c reduced by 10C."
        """
        self.logger.warning("[DeadlockManager] Resolving deadlock for %s | reason=%s", cycle_dir.name, reason)

        crisis_options = self.generate_crisis_options(reason=reason, cycle_dir=cycle_dir)
        votes = self.collect_agent_votes(options=crisis_options, cycle_dir=cycle_dir, reason=reason)

        winning_option_id, tally, tie_broken_by_chef, chef_tiebreak_vote = self._tally_votes(
            votes=votes, crisis_options=crisis_options, cycle_dir=cycle_dir, reason=reason
        )
        winning_option = self._find_option(crisis_options, winning_option_id)

        action = self._classify_action(winning_option.description)

        resolution = DeadlockResolution(
            option_id=winning_option.id,
            description=winning_option.description,
            action=action,
        )

        self._write_cycle_protocol(
            cycle_dir=cycle_dir,
            reason=reason,
            crisis_options=crisis_options,
            votes=votes,
            tally=tally,
            winning_option=winning_option,
            tie_broken_by_chef=tie_broken_by_chef,
            chef_tiebreak_vote=chef_tiebreak_vote,
        )
        self._write_shadow_memory(
            cycle_dir=cycle_dir,
            reason=reason,
            crisis_options=crisis_options,
            votes=votes,
            tally=tally,
            winning_option=winning_option,
            tie_broken_by_chef=tie_broken_by_chef,
            chef_tiebreak_vote=chef_tiebreak_vote,
        )

        self.logger.warning("[DeadlockManager] Deadlock resolved for %s -> %s", cycle_dir.name, resolution)
        return resolution

    def _classify_action(self, description: str) -> str:
        desc_lower = description.lower()

        terminate_keywords = [
            "terminate", "accept the current results", "final report",
            "close", "conclude", "project complete", "goal achieved",
            "halt", "stop", "end the project",
        ]
        if any(kw in desc_lower for kw in terminate_keywords):
            return "terminate_project"

        sim_only_keywords = [
            "simulation-only", "computational task", "without any physical",
            "re-calibrate the digital twin", "update the simulation",
            "in silico", "purely computational",
        ]
        if any(kw in desc_lower for kw in sim_only_keywords):
            return "simulation_only_cycle"

        retry_keywords = [
            "retry with", "adjust and retry", "reduce", "increase",
            "modify parameters",
        ]
        if any(kw in desc_lower for kw in retry_keywords):
            return "retry_with_adjustment"

        return "continue_next_cycle"

    def _tally_votes(
        self,
        votes: dict[str, VoteModel],
        crisis_options: CrisisOptionsModel,
        cycle_dir: Path,
        reason: str,
    ) -> tuple[str, Counter, bool, VoteModel | None]:
        """
        Tally the 3 lead-agent votes and determine a winner.

        Resolution rules:
            - Simple majority (2 or 3 of the 3 votes) wins outright.
            - A genuine 3-way tie (all three agents chose DIFFERENT options)
              is broken by asking ChefAgent to cast one additional, dedicated
              tie-breaking vote. If that tie-break vote ALSO cannot be
              obtained for any reason, we fall back to the deterministic
              default of Option_A - wrapped in its own try/except
              specifically so a failure here can never prevent
              resolve_deadlock() from completing.

        Returns:
            (winning_option_id, tally_counter, tie_broken_by_chef, chef_tiebreak_vote)
            `chef_tiebreak_vote` is None unless a tie-break was actually needed.
        """
        tally: Counter = Counter(vote.vote for vote in votes.values())
        most_common = tally.most_common()

        # A genuine 3-way tie means all three counts are 1 (three distinct
        # options chosen). Any other outcome (2-1 split, or a unanimous 3-0)
        # already has a clear majority via most_common()[0].
        is_three_way_tie = len(most_common) == 3 and all(count == 1 for _, count in most_common)

        if not is_three_way_tie:
            winning_option_id = most_common[0][0]
            return winning_option_id, tally, False, None

        self.logger.warning(
            "[DeadlockManager] 3-way tie detected among lead-agent votes for %s - invoking ChefAgent as tie-breaker.",
            cycle_dir.name,
        )

        chef_tiebreak_vote: VoteModel | None = None
        try:
            chef_tiebreak_vote = self._collect_chef_tiebreak_vote(
                crisis_options=crisis_options, cycle_dir=cycle_dir, reason=reason
            )
            winning_option_id = chef_tiebreak_vote.vote
            self.logger.warning(
                "[DeadlockManager] ChefAgent tie-break vote for %s: %s.", cycle_dir.name, winning_option_id
            )
        except Exception:
            # Complete fallback safety: even the tie-breaker failing must
            # never prevent resolve_deadlock() from completing. Fall back to
            # the deterministic Option_A default.
            self.logger.exception(
                "[DeadlockManager] ChefAgent tie-break vote failed for %s - falling back to Option_A.",
                cycle_dir.name,
            )
            winning_option_id = "Option_A"

        return winning_option_id, tally, True, chef_tiebreak_vote

    def _collect_chef_tiebreak_vote(
        self,
        crisis_options: CrisisOptionsModel,
        cycle_dir: Path,
        reason: str,
    ) -> VoteModel:
        """
        Ask ChefAgent to cast one dedicated tie-breaking vote, using the
        exact same VoteModel schema and prompt shape as the three lead-agent
        votes, framed explicitly as a tie-break so ChefAgent understands the
        stakes of this specific call.
        """
        options_block = self._format_options_block(crisis_options)
        prompt = f"""The lead agents' votes on this crisis resulted in an exact 3-way tie. As ChefAgent,
cast the deciding tie-breaking vote.

## Crisis Reason
{reason}

## Proposed Recovery Options
{options_block}

## Your task
Reason step-by-step about the trade-offs of each option (this is your `analysis`), then
select EXACTLY ONE option by its ID as your `vote` (Option_A, Option_B, or Option_C) to
break the tie.
"""
        system_prompt = (
            "You are ChefAgent, casting the deciding tie-breaking vote after the lab's three "
            "lead agents each chose a different recovery option. Weigh the overall best interest "
            "of the lab's directive and reason carefully before casting the deciding vote."
        )

        chef = ChefAgent(workspace_path=self.workspace_path, model=self.model)
        vote = chef.ask_llm(prompt=prompt, system_prompt=system_prompt, response_model=VoteModel)
        assert isinstance(vote, VoteModel)
        return vote

    def _find_option(self, crisis_options: CrisisOptionsModel, option_id: str) -> OptionModel:
        """Look up an OptionModel by its ID. Guaranteed to exist by construction."""
        for option in crisis_options.options:
            if option.id == option_id:
                return option
        # Unreachable given CrisisOptionsModel's validator guarantees exactly
        # OPTION_IDS are present, and _tally_votes only ever produces an ID
        # from that same fixed set - kept as an explicit, loud failure rather
        # than a silent None/IndexError if that invariant is ever violated by
        # a future change.
        raise ValueError(f"Option ID {option_id!r} not found among crisis options - this should be unreachable.")

    # ------------------------------------------------------------------ #
    # Output persistence
    # ------------------------------------------------------------------ #

    def _write_cycle_protocol(
        self,
        cycle_dir: Path,
        reason: str,
        crisis_options: CrisisOptionsModel,
        votes: dict[str, VoteModel],
        tally: Counter,
        winning_option: OptionModel,
        tie_broken_by_chef: bool,
        chef_tiebreak_vote: VoteModel | None,
    ) -> None:
        """
        Write the human-readable crisis protocol: the reason, the proposed
        options, the full voting breakdown with each agent's chain-of-thought
        analysis, the tally, and the final resolution.
        """
        analysis_dir = cycle_dir / "C_Analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        options_section = "\n".join(f"- **{option.id}**: {option.description}" for option in crisis_options.options)

        votes_section = "\n\n".join(
            f"### {agent_name} -> {vote.vote}\n{vote.analysis}" for agent_name, vote in votes.items()
        )

        tally_section = "\n".join(f"- {option_id}: {count} vote(s)" for option_id, count in tally.most_common())

        tiebreak_section = ""
        if tie_broken_by_chef:
            if chef_tiebreak_vote is not None:
                tiebreak_section = (
                    f"\n## Tie-Break (ChefAgent)\nA 3-way tie occurred among the lead-agent votes. "
                    f"ChefAgent cast the deciding vote for **{chef_tiebreak_vote.vote}**.\n\n"
                    f"{chef_tiebreak_vote.analysis}\n"
                )
            else:
                tiebreak_section = (
                    "\n## Tie-Break (Fallback)\nA 3-way tie occurred among the lead-agent votes, and "
                    "the ChefAgent tie-breaking vote could not be obtained. Defaulted to **Option_A** "
                    "per the deadlock manager's fallback safety rule.\n"
                )

        report = f"""# Cycle Protocol - Deadlock Resolution

## Crisis Reason
{reason}

## Proposed Recovery Options
{options_section}

## Voting Breakdown (Chain-of-Thought)
{votes_section}

## Tally
{tally_section}
{tiebreak_section}
## Resolution
**{winning_option.id}**: {winning_option.description}
"""
        protocol_path = analysis_dir / CYCLE_PROTOCOL_FILENAME
        protocol_path.write_text(report, encoding="utf-8")
        self.logger.info("[DeadlockManager] Wrote cycle protocol to %s", protocol_path)

    def _write_shadow_memory(
        self,
        cycle_dir: Path,
        reason: str,
        crisis_options: CrisisOptionsModel,
        votes: dict[str, VoteModel],
        tally: Counter,
        winning_option: OptionModel,
        tie_broken_by_chef: bool,
        chef_tiebreak_vote: VoteModel | None,
    ) -> None:
        """
        Persist the full, machine-readable audit trail: the crisis reason,
        every proposed option, every agent's full vote (analysis + choice),
        the tally, whether a tie-break was needed, and the final winner.
        """
        shadow_dir = cycle_dir / "D_Shadow_Memory"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        shadow_record = {
            "component": "DeadlockManager",
            "model": self.model,
            "reason": reason,
            "crisis_options": crisis_options.model_dump(),
            "votes": {agent_name: vote.model_dump() for agent_name, vote in votes.items()},
            "tally": dict(tally),
            "tie_broken_by_chef": tie_broken_by_chef,
            "chef_tiebreak_vote": chef_tiebreak_vote.model_dump() if chef_tiebreak_vote is not None else None,
            "winning_option": winning_option.model_dump(),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }

        shadow_path = shadow_dir / DEADLOCK_SHADOW_FILENAME
        shadow_path.write_text(json.dumps(shadow_record, indent=2), encoding="utf-8")
        self.logger.info("[DeadlockManager] Wrote shadow memory to %s", shadow_path)
