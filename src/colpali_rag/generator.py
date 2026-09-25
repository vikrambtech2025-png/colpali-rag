"""Pluggable answer generator.

- TextGenerator: retrieves the best page text (stored at ingest time) and asks a
  free-tier kilo LLM on the local OmniRoute gateway (OpenAI-compatible). 100%
  token-free pipeline: embeddings, ColPali and OCR are all local.
- ExtractiveGenerator: local no-LLM answerer that pulls the most query-relevant
  sentences straight out of the retrieved page text. Zero dependencies.
- VisionGenerator: answers from the actual page images (charts, tables, layouts)
  via a vision-capable model on the OmniRoute gateway.
- generate_with_fallback(): runs the configured generator and, when it produces
  no answer (unreachable gateway / unimplemented backend), falls back to the
  extractive answerer so retrieval always returns something useful.
"""
from __future__ import annotations

import base64
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

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
        "Cite pages inline by copying the exact bracketed source label verbatim, e.g. "
        "[Q3-2025-Market-Intelligence-Report.pdf (page 1)] — do not invent or shorten the label.\n\n"
        "EXCERPTS:\n{excerpts}\n\n"
        "QUESTION: {question}\n\nANSWER:"
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, query: str, pages: list[PageResult]) -> Generation:
        from openai import OpenAI

        excerpts = build_excerpt(pages, self.settings.generation_top_pages, self.settings.excerpt_chars)
        prompt = self.PROMPT.format(excerpts=excerpts, question=query)
        attempts = max(1, self.settings.generation_retries + 1)
        last_reason = ""
        for attempt in range(attempts):
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
                if answer:
                    return Generation(
                        answer=answer,
                        model=self.settings.omniroute_model,
                        backend="text",
                        citations=self.citations_for(pages, self.settings.generation_top_pages),
                    )
                last_reason = "the gateway returned an empty reply"
                log.warning("Generation attempt %d: %s", attempt + 1, last_reason)
            except Exception as exc:  # keep retrieval usable even if the gateway is down
                last_reason = str(exc) or type(exc).__name__
                low = last_reason.lower()
                if "sign in" in low or "authentication" in low or "expired" in low:
                    last_reason = (
                        f"gateway provider auth problem (likely needs re-sign-in in the "
                        f"OmniRoute dashboard): {last_reason}"
                    )
                log.warning("Generation attempt %d failed: %s", attempt + 1, last_reason)
            if attempt < attempts - 1:
                time.sleep(self.settings.generation_retry_delay)
        return Generation(
            answer="",
            backend="text",
            note=f"generation backend unreachable ({attempts} attempt(s)): {last_reason}",
            citations=self.citations_for(pages, self.settings.generation_top_pages),
        )


class VisionGenerator(Generator):
    """Answer from the actual page images via a vision-capable gateway model.

    The top retrieved pages' rendered PNGs (data/pages/<src>.pdf/page-NNN.png)
    are base64 data URIs in an OpenAI-style multimodal message; the model reads
    the true layout - charts, tables, axes - that text extraction can mangle.
    Falls back through generate_with_fallback() when the gateway is down.
    """

    name = "vision"

    PROMPT = (
        "You are an exact-answer assistant that reads document pages from images.\n"
        "Answer the question using ONLY the attached page images (they show the exact "
        "charts, tables and layouts). Quote numbers, axis labels and ranges precisely. "
        "If the images cannot answer it, say so. Cite pages inline by copying the exact "
        "bracketed source label verbatim, e.g. "
        "[NanoSat-Constellation-Design-Spec.pdf (page 3)] - do not invent or shorten it.\n\n"
        "QUESTION: {question}\n\nANSWER:"
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, query: str, pages: list[PageResult]) -> Generation:
        from openai import OpenAI

        model = self.settings.omniroute_vision_model
        citations = self.citations_for(pages, self.settings.generation_top_pages)
        content: list[dict[str, Any]] = [{"type": "text", "text": self.PROMPT.format(question=query)}]
        images = 0
        for p in pages[: self.settings.generation_top_pages]:
            data_uri = self._image_data_uri(p)
            if data_uri is None:
                continue
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
            images += 1
        if images == 0:
            return Generation(
                answer="",
                backend=self.name,
                note="no page images available to feed the vision model",
                citations=citations,
            )
        attempts = max(1, self.settings.generation_retries + 1)
        last_reason = ""
        for attempt in range(attempts):
            try:
                client = OpenAI(
                    base_url=self.settings.omniroute_base_url,
                    api_key=self.settings.omniroute_api_key or "local",
                    timeout=self.settings.llm_timeout,
                )
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You answer from document images only."},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    stream=False,
                )
                answer = (resp.choices[0].message.content or "").strip()
                if answer:
                    return Generation(
                        answer=answer,
                        model=model,
                        backend="vision",
                        citations=citations,
                    )
                last_reason = "the gateway returned an empty reply"
                log.warning("Vision generation attempt %d: %s", attempt + 1, last_reason)
            except Exception as exc:  # keep retrieval usable even if the gateway is down
                last_reason = str(exc) or type(exc).__name__
                low = last_reason.lower()
                if "sign in" in low or "authentication" in low or "expired" in low:
                    last_reason = (
                        f"gateway provider auth problem (likely needs re-sign-in in the "
                        f"OmniRoute dashboard): {last_reason}"
                    )
                log.warning("Vision generation attempt %d failed: %s", attempt + 1, last_reason)
            if attempt < attempts - 1:
                time.sleep(self.settings.generation_retry_delay)
        return Generation(
            answer="",
            backend=self.name,
            note=f"vision backend unreachable ({attempts} attempt(s)): {last_reason}",
            citations=citations,
        )

    def _image_data_uri(self, page: PageResult) -> str | None:
        try:
            path = self.settings.pages_dir / page.image
            if not path.is_file():
                log.debug("page image missing: %s", path)
                return None
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{b64}"
        except Exception as exc:  # pragma: no cover
            log.debug("could not read page image: %s", exc)
            return None


class ExtractiveGenerator(Generator):
    """No-LLM answerer: scores sentences in the retrieved page text by lexical
    overlap with the query and returns the best ones with citations. Always
    available locally - works with the gateway down and adds no latency."""

    name = "extractive"

    _TOKEN_RE = re.compile(r"[a-z0-9]+")
    _SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
    _STOPS = frozenset(
        "a an the is are was were be been being of to in for and or on with at by from as "
        "it its this that these those what which who whom how when where why do does did "
        "has have had not no so if then than but you your we our they their".split()
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    @property
    def _top_pages(self) -> int:
        return self.settings.generation_top_pages if self.settings else 3

    def _tokens(self, text: str) -> list[str]:
        return [t for t in self._TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in self._STOPS]

    def _sentences(self, text: str) -> list[str]:
        return [s.strip() for s in self._SENT_RE.split(text or "") if len(s.strip()) >= 10]

    def generate(self, query: str, pages: list[PageResult]) -> Generation:
        citations = self.citations_for(pages, self._top_pages)
        q_tokens = set(self._tokens(query))
        if not q_tokens:
            return Generation(
                answer="", backend=self.name,
                note="query has no searchable tokens", citations=citations,
            )
        scored: list[tuple[float, str, str]] = []
        for p in pages[: self._top_pages]:
            best_score, best_sent = 0.0, ""
            for sent in self._sentences(p.text):
                toks = self._tokens(sent)
                if not toks:
                    continue
                hits = sum(1 for t in toks if t in q_tokens)
                if hits == 0:
                    continue
                score = hits / math.sqrt(len(toks))  # normalize against sentence length
                if score > best_score:
                    best_score, best_sent = score, sent
            if best_sent:
                scored.append((best_score, p.citation, best_sent))
        scored.sort(key=lambda x: -x[0])
        seen: set[str] = set()
        parts: list[str] = []
        for _score, cit, sent in scored:
            key = sent.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"[{cit}] {sent}")
        if not parts:
            return Generation(
                answer="No directly matching text was found in the retrieved pages.",
                backend=self.name, citations=citations,
            )
        answer = " ".join(parts)
        if len(answer) > 900:
            answer = answer[:900].rsplit(" ", 1)[0] + " ..."
        return Generation(answer=answer, backend=self.name, citations=citations)


def generate_with_fallback(generator: Generator, query: str, pages: list[PageResult]) -> Generation:
    """Run the configured generator; if it yields no answer (gateway unreachable,
    unimplemented backend, exception), answer extraction-locally instead so the
    API always returns a usable answer when pages were retrieved."""
    if isinstance(generator, ExtractiveGenerator):
        return generator.generate(query, pages)
    try:
        result = generator.generate(query, pages)
    except Exception as exc:  # e.g. VisionGenerator stub raising NotImplementedError
        result = Generation(answer="", backend=generator.name, note=f"{generator.name} generator failed: {exc}")
    if result.answer:
        return result
    fallback = ExtractiveGenerator(getattr(generator, "settings", None))
    fb = fallback.generate(query, pages)
    if fb.answer:
        reason = result.note or f"{generator.name} produced no answer"
        fb.note = f"{reason} | fallback: extractive"
        return fb
    return result


def get_generator(settings: Settings) -> Generator:
    if settings.generation_mode == "vision":
        return VisionGenerator(settings)
    if settings.generation_mode == "extractive":
        return ExtractiveGenerator(settings)
    return TextGenerator(settings)