"""
Abstract base class for all pipeline agents.
Provides:
  - Structured logging (self.logger)
  - Unified LLM query interface with retries (self.query)
  - JSON parsing helpers
"""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import openai

from ..config import PipelineConfig
from ..monitoring.logger import PipelineLogger


class BaseAgent(ABC):
    """
    All pipeline agents inherit from this class.

    Subclasses only need to implement their domain logic.
    LLM calls are routed through self.query() which handles:
      - Backend selection (voyager, vllm, openai)
      - Exponential backoff retries
      - <think> block stripping
      - JSON fallback extraction
    """

    def __init__(self, config: PipelineConfig, name: str):
        self.config = config
        self.name = name
        self.logger = PipelineLogger(name, level=config.log_level, fmt=config.log_format)

    # ── LLM interface ─────────────────────────────────────────────────────────

    def query(
        self,
        prompt: str,
        system_prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.05,
        retries: int = 3,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Unified LLM query with exponential backoff.

        Returns the model's text response (thinking blocks stripped).
        Raises RuntimeError after exhausting retries.
        """
        model = model or self.config.doctor_model
        backend = self.config.model_backend
        if timeout is None:
            timeout = self.config.request_timeout

        for attempt in range(retries):
            try:
                response = self._call_backend(
                    backend=backend,
                    model=model,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
                return self._strip_thinking(response)

            except Exception as exc:
                wait = 2 ** attempt
                self.logger.warning(
                    "llm_retry",
                    data={
                        "attempt": attempt + 1,
                        "error": str(exc),
                        "wait_seconds": wait,
                        "model": model,
                    },
                )
                time.sleep(wait)

        raise RuntimeError(
            f"[{self.name}] LLM query failed after {retries} retries."
        )

    def _call_backend(
        self,
        backend: str,
        model: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        if backend in ("voyager", "vllm", "openai"):
            saved_base = openai.api_base
            saved_key = openai.api_key
            try:
                if self.config.api_base_url:
                    openai.api_base = self.config.api_base_url
                if self.config.api_key:
                    openai.api_key = self.config.api_key

                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=timeout,
                )
                return response["choices"][0]["message"]["content"]
            finally:
                openai.api_base = saved_base
                openai.api_key = saved_key
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>…</think> blocks emitted by reasoning models."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # ── JSON parsing helpers ──────────────────────────────────────────────────

    @staticmethod
    def parse_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse JSON from model output.
        1. Direct json.loads
        2. Extract first {...} block via regex
        3. Return None on failure
        """
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to extract a JSON object from surrounding text
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def parse_json_with_salvage(text: str) -> Optional[Dict[str, Any]]:
        """
        Extended JSON parser that handles truncated model output.

        When standard parsing fails (e.g. response cut off mid-array), this
        scans the raw text and tries to parse every ``{...}`` block it finds,
        at any nesting depth.  Any complete block that looks like a hypothesis
        (has "disease" and "confidence" keys) is collected and returned as:
            {"hypotheses": [<complete objects only>]}

        This recovers whatever hypotheses were fully generated even when the
        last one (or the outer wrapper) was truncated by the token limit.
        """
        # Fast path: standard parse succeeds
        result = BaseAgent.parse_json(text)
        if result:
            return result

        # Salvage: for every '{' in the text, find its matching '}' and try
        # to parse that slice.  We do NOT rely on nesting depth reaching 0
        # globally (which would require the outer object to be complete).
        objects = []
        n = len(text)
        i = 0
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            # Walk forward to find the balanced closing '}'
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                # Found a balanced block — try to parse it
                candidate = text[i:j]
                try:
                    obj = json.loads(candidate)
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                i = j  # skip past this entire block
            else:
                # Unbalanced (truncated) — skip this '{' and keep looking
                i += 1

        # Keep only hypothesis-like objects (must have "disease" + "confidence")
        hyp_objects = [o for o in objects if "disease" in o and "confidence" in o]
        if hyp_objects:
            return {"hypotheses": hyp_objects}

        return None

    @staticmethod
    def extract_field(text: str, field: str) -> Optional[str]:
        """
        Regex fallback: extract a simple field value from non-JSON model output.
        Example: extract_field(text, "diagnosis") looks for 'diagnosis: ...'
        """
        pattern = rf'"{field}"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        # Plain-text fallback: "FIELD: value"
        pattern2 = rf"{re.escape(field)}\s*[:=]\s*(.+?)(?:\n|$)"
        match2 = re.search(pattern2, text, re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        return None
