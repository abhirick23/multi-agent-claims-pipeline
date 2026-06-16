"""Thin wrapper around the Gemini API for vision-based document classification/extraction and
LLM-assisted canonical mapping.

All structured outputs use ``response_schema`` so responses validate directly against Pydantic
models. Calls retry with exponential backoff on transient/rate-limit errors; once retries are
exhausted, :class:`~app.core.exceptions.GeminiAPIError` is raised so the calling agent can degrade
gracefully instead of crashing the pipeline.

In "injection mode" (test harness), agents check ``DocumentInput`` for pre-supplied data and never
construct/call this client at all -- it is only exercised in the live Streamlit demo.
"""
from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from google import genai

load_dotenv()
from google.genai import errors, types
from pydantic import BaseModel

from app.core.exceptions import GeminiAPIError
from app.core.logging_config import get_logger

_log = get_logger(__name__)
from app.llm.prompts import (
    CANONICAL_MAPPING_PROMPT,
    CLASSIFY_DOCUMENT_PROMPT,
    EXTRACT_CONTENT_PROMPT,
)
from app.llm.schemas import GeminiCanonicalMappingResponse, GeminiDocumentClassification
from app.models.common import DocumentType
from app.models.extraction import ExtractedContent

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2.0

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GeminiAPIError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    def classify_document(self, image_path: str) -> GeminiDocumentClassification:
        _log.info("[GEMINI] classify_document START — model=%s, path=%s", self._model, image_path)
        t0 = time.monotonic()
        image_part = self._load_image(image_path)
        result = self._generate_structured(
            contents=[image_part, CLASSIFY_DOCUMENT_PROMPT],
            response_schema=GeminiDocumentClassification,
            call_name="classify_document",
        )
        _log.info("[GEMINI] classify_document END — %.2fs → type=%s quality=%s", time.monotonic() - t0, result.document_type, result.quality)
        return result

    def extract_content(self, image_path: str, document_type: DocumentType) -> ExtractedContent:
        _log.info("[GEMINI] extract_content START — model=%s, doc_type=%s, path=%s", self._model, document_type.value, image_path)
        t0 = time.monotonic()
        image_part = self._load_image(image_path)
        prompt = EXTRACT_CONTENT_PROMPT.format(document_type=document_type.value)
        result = self._generate_structured(
            contents=[image_part, prompt],
            response_schema=ExtractedContent,
            call_name="extract_content",
        )
        _log.info("[GEMINI] extract_content END — %.2fs → patient=%s total=%s", time.monotonic() - t0, result.patient_name, result.total)
        return result

    def classify_canonical_category(
        self,
        diagnosis_text: str,
        treatment_text: str,
        waiting_period_keys: list[str],
        exclusion_terms: list[str],
    ) -> GeminiCanonicalMappingResponse:
        _log.info("[GEMINI] classify_canonical_category START — diagnosis=%r treatment=%r", diagnosis_text, treatment_text)
        t0 = time.monotonic()
        prompt = CANONICAL_MAPPING_PROMPT.format(
            diagnosis_text=diagnosis_text or "",
            treatment_text=treatment_text or "",
            waiting_period_keys=waiting_period_keys,
            exclusion_terms=exclusion_terms,
        )
        result = self._generate_structured(
            contents=[prompt],
            response_schema=GeminiCanonicalMappingResponse,
            call_name="classify_canonical_category",
        )
        _log.info("[GEMINI] classify_canonical_category END — %.2fs → waiting_period_key=%s exclusions=%d", time.monotonic() - t0, result.waiting_period_key, len(result.matched_exclusion_terms))
        return result

    def _load_image(self, image_path: str) -> types.Part:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(path.name)
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type or "image/jpeg")

    def _generate_structured(self, contents: list, response_schema: type[ResponseT], call_name: str = "gemini") -> ResponseT:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
                return response_schema.model_validate_json(response.text)
            except errors.APIError as exc:
                last_error = exc
                wait = self._backoff_seconds * (2 ** attempt)
                if attempt < self._max_retries - 1:
                    _log.warning("[GEMINI] %s attempt %d/%d failed (APIError: %s) — retrying in %.1fs", call_name, attempt + 1, self._max_retries, exc, wait)
                    time.sleep(wait)
                else:
                    _log.error("[GEMINI] %s all %d attempts failed — APIError: %s", call_name, self._max_retries, exc)
            except Exception as exc:  # malformed/unvalidatable response
                last_error = exc
                wait = self._backoff_seconds * (2 ** attempt)
                if attempt < self._max_retries - 1:
                    _log.warning("[GEMINI] %s attempt %d/%d failed (%s: %s) — retrying in %.1fs", call_name, attempt + 1, self._max_retries, type(exc).__name__, exc, wait)
                    time.sleep(wait)
                else:
                    _log.error("[GEMINI] %s all %d attempts failed — %s: %s", call_name, self._max_retries, type(exc).__name__, exc)
        raise GeminiAPIError(f"Gemini call failed after {self._max_retries} attempts: {last_error}")
