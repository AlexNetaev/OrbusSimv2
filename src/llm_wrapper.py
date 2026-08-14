"""
llm_wrapper.py
================
Robust wrapper around the local Ollama client that gives every agent in the
Self-Driving Lab two guarantees other code can rely on without re-implementing
error handling every time:

  1. Plain-text calls never silently return an empty/garbage response - they
     retry until they get real content (or exhaust retries).
  2. Structured calls (response_model=SomePydanticModel) ALWAYS either return
     a validated instance of that model, or raise LLMValidationException.
     Callers never have to sprinkle try/except json.JSONDecodeError all over
     the agent codebase.

The "auto-correction" loop is the key trick: when the model returns invalid
JSON, we don't just retry blindly - we feed the exact validation error back
into the next prompt so the model can see what it got wrong and fix it. This
dramatically improves structured-output reliability with local, smaller models
that don't have native tool-calling/JSON-mode guarantees.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Type, TypeVar, overload

import ollama
from pydantic import BaseModel, ValidationError

from src import config

logger = logging.getLogger(__name__)

# Generic type var so callers get proper static typing back:
# ask_llm_with_validation(..., response_model=Measurement) -> Measurement
T = TypeVar("T", bound=BaseModel)


class LLMValidationException(Exception):
    """
    Raised when the LLM fails to produce output that satisfies the requested
    Pydantic schema after `max_retries` attempts.

    Carries the last raw response and last error so the caller (or a human
    debugging logs) can see exactly what went wrong on the final attempt.
    """

    def __init__(self, message: str, last_raw_response: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_raw_response = last_raw_response
        self.last_error = last_error


# --------------------------------------------------------------------------- #
# Overloads purely for static-typing ergonomics: callers get `str` back when
# response_model is omitted, and `T` back when it's provided. This has no
# runtime effect - the real implementation follows below.
# --------------------------------------------------------------------------- #
@overload
def ask_llm_with_validation(
    prompt: str,
    system_prompt: str = "",
    response_model: None = None,
    max_retries: int = config.DEFAULT_MAX_RETRIES,
    model: str = config.OLLAMA_MODEL,
    temperature: float = config.DEFAULT_TEMPERATURE,
    context_size: int | None = None,
) -> str: ...


@overload
def ask_llm_with_validation(
    prompt: str,
    system_prompt: str = "",
    response_model: Type[T] = ...,
    max_retries: int = config.DEFAULT_MAX_RETRIES,
    model: str = config.OLLAMA_MODEL,
    temperature: float = config.DEFAULT_TEMPERATURE,
    context_size: int | None = None,
) -> T: ...


def ask_llm_with_validation(
    prompt: str,
    system_prompt: str = "",
    response_model: Type[BaseModel] | None = None,
    max_retries: int = config.DEFAULT_MAX_RETRIES,
    model: str = config.OLLAMA_MODEL,
    temperature: float = config.DEFAULT_TEMPERATURE,
    context_size: int | None = None,
) -> str | BaseModel:
    """
    Ask the local Ollama model a question, with automatic retry + self-correction.

    Args:
        prompt: The user-facing prompt / task content.
        system_prompt: System-level instructions (persona, constraints, etc).
                       When response_model is provided, the model's JSON schema
                       is automatically appended to this.
        response_model: Optional Pydantic model class. If provided, the raw
                         text response is parsed & validated against it, with
                         retries-with-error-feedback on failure.
        max_retries: Maximum number of attempts before giving up.
        model: Which Ollama model tag to call. Defaults to config.OLLAMA_MODEL.
        temperature: Sampling temperature passed to Ollama.
        context_size: Optional context window size (Ollama's `num_ctx`).
                       None leaves it unset (Ollama's own default applies) -
                       see `BaseAgent.ask_llm`'s Step 8 UI-settings
                       integration for where a concrete value normally comes
                       from.

    Returns:
        - If response_model is None: the raw response string (guaranteed non-empty).
        - If response_model is provided: a validated instance of that model.

    Raises:
        LLMValidationException: if response_model was provided and validation
                                  still fails after max_retries attempts.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    # Build the effective system prompt. If a response_model was supplied, we
    # inject its JSON schema directly into the system prompt so the model has
    # an explicit contract to follow, rather than guessing field names/types.
    effective_system_prompt = system_prompt
    if response_model is not None:
        effective_system_prompt = _build_structured_system_prompt(system_prompt, response_model)

    current_prompt = prompt
    last_raw_response = ""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        logger.debug("LLM attempt %d/%d (model=%s)", attempt, max_retries, model)

        raw_response = _call_ollama(
            prompt=current_prompt,
            system_prompt=effective_system_prompt,
            model=model,
            temperature=temperature,
            context_size=context_size,
        )
        last_raw_response = raw_response

        # --- Case 1: plain-text mode (no schema requested) ------------------
        if response_model is None:
            if raw_response.strip():
                return raw_response.strip()

            # Empty / EOS-only response: retry silently with the same prompt.
            # We deliberately do NOT append error text to the prompt here,
            # since there's no "correction" to make - it's usually just a
            # flaky/empty generation, and re-asking as-is is the right fix.
            logger.warning("Empty response from LLM on attempt %d/%d - retrying.", attempt, max_retries)
            continue

        # --- Case 2: structured/validated mode -------------------------------
        try:
            cleaned = _strip_markdown_json_fence(raw_response)
            if not cleaned.strip():
                raise ValueError("Model returned an empty response where JSON was expected.")

            validated_instance = response_model.model_validate_json(cleaned)
            return validated_instance

        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Structured output validation failed on attempt %d/%d: %s",
                attempt,
                max_retries,
                exc,
            )

            # Auto-correction: feed the exact error back to the model so the
            # NEXT attempt can see precisely what was wrong and fix it. We
            # rebuild the prompt (rather than mutate the original) so each
            # retry includes full context: original task + last bad output +
            # the error that resulted from it.
            current_prompt = _build_correction_prompt(
                original_prompt=prompt,
                bad_response=raw_response,
                error=exc,
            )
            continue

    # If we fall out of the loop, every attempt failed.
    if response_model is None:
        # Ran out of retries trying to get a non-empty plain-text response.
        raise LLMValidationException(
            message=f"LLM returned only empty responses after {max_retries} attempts.",
            last_raw_response=last_raw_response,
            last_error=last_error,
        )

    raise LLMValidationException(
        message=(
            f"LLM failed to produce output matching schema "
            f"'{response_model.__name__}' after {max_retries} attempts."
        ),
        last_raw_response=last_raw_response,
        last_error=last_error,
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _call_ollama(
    prompt: str, system_prompt: str, model: str, temperature: float, context_size: int | None = None
) -> str:
    """
    Make a single call to the Ollama chat endpoint and return the raw text
    content. Isolated into its own function so retries/mocking are simple.

    `context_size`, when provided, is passed through as Ollama's `num_ctx`
    option (the model's context window size in tokens). Optional and
    defaulted to None (letting Ollama use its own built-in default) rather
    than a hardcoded value here, since the effective default is
    `config.DEFAULT_CONTEXT_SIZE` - resolved by the caller (see
    `BaseAgent.ask_llm`'s Step 8 UI-settings integration), not baked into
    this low-level function.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    options: dict = {"temperature": temperature}
    if context_size is not None:
        options["num_ctx"] = context_size

    response = ollama.chat(
        model=model,
        messages=messages,
        options=options,
    )

    # ollama-python returns a ChatResponse (dict-like) with response["message"]["content"]
    try:
        return response["message"]["content"] or ""
    except (KeyError, TypeError):
        # Defensive fallback in case the client's return shape changes between
        # versions - we'd rather return "" (triggering a retry) than crash.
        logger.error("Unexpected Ollama response shape: %r", response)
        return ""


def _build_structured_system_prompt(base_system_prompt: str, response_model: Type[BaseModel]) -> str:
    """
    Append a strict JSON-only instruction plus the target Pydantic model's
    JSON schema to the system prompt, so the model has an explicit contract.
    """
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)

    structured_instructions = f"""
You must respond with STRICTLY VALID JSON that conforms exactly to the following
JSON Schema. Do not include any explanation, markdown formatting, code fences,
or conversational text before or after the JSON - output ONLY the raw JSON object.

JSON Schema:
{schema_json}
""".strip()

    if base_system_prompt.strip():
        return f"{base_system_prompt.strip()}\n\n{structured_instructions}"
    return structured_instructions


def _build_correction_prompt(original_prompt: str, bad_response: str, error: Exception) -> str:
    """
    Build a retry prompt that shows the model exactly what it produced and
    exactly what was wrong with it, so it can self-correct instead of
    repeating the same mistake.
    """
    return f"""{original_prompt}

Your previous response was NOT valid according to the required JSON schema.

Your previous output was:
---
{bad_response}
---

The validation error was:
---
{error}
---

Fix the JSON so it strictly and completely matches the required schema.
Return ONLY the corrected JSON object - no explanation, no markdown code fences,
no conversational text.""".strip()


def _strip_markdown_json_fence(text: str) -> str:
    """
    Local models frequently wrap JSON in markdown code fences even when told
    not to (e.g. ```json { ... } ``` or plain ``` { ... } ```). Strip those
    fences (and any leading/trailing prose outside the outermost {} or [])
    before attempting to parse.
    """
    text = text.strip()

    # Remove ```json ... ``` or ``` ... ``` fences.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # As a further safety net, if there's still leading/trailing junk text
    # around the JSON body, extract the outermost {...} or [...] block.
    # This handles cases like: "Sure! Here's the JSON: {...} Let me know!"
    if not (text.startswith("{") or text.startswith("[")):
        obj_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if obj_match:
            text = obj_match.group(1).strip()

    return text


# --------------------------------------------------------------------------- #
# Self-test / demonstration.
#
# This block intentionally gives the model a MESSY, unstructured, natural-
# language description of a lab measurement and forces it through the
# validation pipeline into a strict Pydantic model. It exercises:
#   - JSON schema injection into the system prompt
#   - markdown-fence stripping
#   - the auto-correction retry loop (small/local models often get types or
#     field names wrong on the first try)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    class ScientificMeasurement(BaseModel):
        """Strict schema we want the messy lab note coerced into."""
        experiment_name: str
        measured_value: float
        unit: str
        instrument: str
        is_within_tolerance: bool

    messy_note = (
        "ok so for the beaker heating trial we got roughly 87.3 degrees C on the "
        "thermocouple probe, that's basically in spec (tolerance was +/- 5C from "
        "target 85C so we're fine), experiment was called 'Beaker Heating Trial 12'"
    )

    print("=" * 70)
    print(f"Testing ask_llm_with_validation() against model: {config.OLLAMA_MODEL}")
    print("=" * 70)

    try:
        result = ask_llm_with_validation(
            prompt=(
                "Extract the structured measurement data from this messy lab note:\n\n"
                f'"{messy_note}"'
            ),
            system_prompt=(
                "You are a meticulous lab data-entry assistant. Extract exactly the "
                "fields requested from the operator's note. Infer is_within_tolerance "
                "from the operator's own statement about tolerance/spec."
            ),
            response_model=ScientificMeasurement,
            model=config.OLLAMA_MODEL,
        )

        print("\n✅ SUCCESS - validated Pydantic instance returned:\n")
        print(result.model_dump_json(indent=2))

    except LLMValidationException as e:
        print("\n❌ FAILED after all retries.")
        print(f"Error: {e}")
        print(f"Last raw response from model:\n{e.last_raw_response}")

    except Exception as e:
        # Most likely cause in a sandboxed/offline environment: no reachable
        # Ollama server at config.OLLAMA_HOST, or the model tag isn't pulled.
        print(f"\n⚠️  Could not complete live test - is Ollama running locally "
              f"with '{config.OLLAMA_MODEL}' pulled?\nUnderlying error: {e}")
