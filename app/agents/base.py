"""Shared base class for pipeline agents.

Agents are plain classes (not a framework) -- each implements a ``run`` method with its own typed
input/output (see ``app/models``). The only shared behaviour is lazy construction of a
:class:`GeminiClient`, so unit tests that exercise injection-mode paths never need a
``GEMINI_API_KEY`` and never hit the network.
"""
from __future__ import annotations

import logging

from app.core.logging_config import get_logger
from app.llm.gemini_client import GeminiClient


class BaseAgent:
    def __init__(self, gemini_client: GeminiClient | None = None):
        self._gemini_client = gemini_client
        self.log: logging.Logger = get_logger(type(self).__module__ + "." + type(self).__name__)

    @property
    def gemini(self) -> GeminiClient:
        if self._gemini_client is None:
            self._gemini_client = GeminiClient()
        return self._gemini_client
