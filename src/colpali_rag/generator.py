"""Pluggable answer generator.

- TextGenerator: retrieves the best page text (stored at ingest time) and asks a
  free-tier kilo LLM on the local OmniRoute gateway (OpenAI-compatible). 100%
  token-free pipeline: embeddings, ColPali and OCR are all local.
- VisionGenerator: stub slot for feeding the actual page image to a vision LLM.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import Settings
from .retrieve import PageResult

log = logging.getLogger(__name__)


@dataclass
class Generation:
    answer: str
    model: str = ""
    backend: str = ""  # text | vision
    note: str = ""
    citations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.answer.strip()) and self.backend != "unconfigured"


class Generator(ABC):
    name = "base"

    @abstractmethod
    def generate(self, query: str, pages: list[PageResult]) -> Generation: ...

    @staticmethod
    def citations_for(pages: list[PageResult], top_pages: int) -> list[str]:
        return [p.citation for p in pages[:top_pages]]


def build_excerpt(pages: list[PageResult], top_pages: int, char_limit: int) -> str:
    """Concatenate page text with clear citations for the LLM context."""
    blocks: list[str] = []
    used = 0
    for p in pages[:top_pages]:
        text = (p.text or "").strip()
        if not text:
            continue
        if used >= char_limit:
            break
        piece = text[: max(0, char_limit - used)]
        blocks.append(f"[{p.citation}]\n{piece}")
        used += len(piece)
    return "\n\n".join(blocks) or "[No extractable text found on retrieved pages.]"


class TextGenerator(Generator):
    """Answer from page text via the OmniRoute gateway (kilo models, free tier)."""

    name = "text"

    PROMPT = (
        "You are an exact-answer assistant grounded strictly in the provided document excerpts.\n"
        "Answer the question using ONLY the excerpts below. If the excerpts cannot answer it, say so.\n"
        "Cite pages inline like [file.pdf (page N)].\n\n"
        "EXCERPTS:\n{excerpts}\n\n"
        "QUESTION: {question}\n\nANSWER:"
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, query: str, pages: list[PageResult]) -> Generation:
        from openai import OpenAI

        excerpts = build_excerpt(pages, self.settings.generation_top_pages, self.settings.excerpt_chars)
        prompt = self.PROMPT.format(excerpts=excerpts, question=query)
        try:
            client = OpenAI(
                base_url=self.settings.omniroute_base_url,
                api_key=self.settings.omniroute_api_key or "local",
                timeout=self.settings.llm_timeout,
            )
            resp = client.chat.completions.create(
                model=self.settings.omniroute_model,
                messages=[
                    {"role": "system", "content": "You answer from documents only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                stream=False,
            )
            answer = (resp.choices[0].message.content or "").strip()
            return Generation(
                answer=answer,
                model=self.settings.omniroute_model,
                backend="text",
                citations=self.citations_for(pages, self.settings.generation_top_pages),
            )
        except Exception as exc:  # keep retrieval usable even if the gateway is down
            log.warning("Generation failed: %s", exc)
            return Generation(
                answer="",
                backend="text",
                note=f"generation backend unreachable: {exc}",
                citations=self.citations_for(pages, self.settings.generation_top_pages),
            )


class VisionGenerator(Generator):
    """Slot for visual answering (feed page image to a vision LLM). NotImplemented until a
    vision model is configured — retrieval + citations still work in the meantime."""

    name = "vision"

    def generate(self, query: str, pages: list[PageResult]) -> Generation:
        raise NotImplementedError(
            "VisionGenerator is a stub: configure a vision-capable LLM to answer from page images."
        )


def get_generator(settings: Settings) -> Generator:
    if settings.generation_mode == "vision":
        return VisionGenerator()
    return TextGenerator(settings)