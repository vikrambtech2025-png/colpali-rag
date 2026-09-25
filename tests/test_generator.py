"""Unit tests for the generator layer (no models, no network)."""
from __future__ import annotations

import pytest

from colpali_rag.config import Settings
from colpali_rag.generator import (
    ExtractiveGenerator,
    Generation,
    TextGenerator,
    VisionGenerator,
    generate_with_fallback,
    get_generator,
)
from colpali_rag.qdrant_store import PageResult


def _page(text: str, src: str = "rpt.pdf", page: int = 1, score: float = 0.9) -> PageResult:
    return PageResult(score=score, src=src, page=page, text=text)


PAGES = [
    _page(
        "Revenue grew to $37M in Q3 2025, led by the Cloud product line (22M USD). "
        "Devices declined slightly while Services stayed flat.",
        "Q3-2025.pdf", 1,
    ),
    _page(
        "Battery chemistry table: LFP costs $45 per kWh at 210 Wh/kg density. "
        "NMC 811 costs $62 per kWh.",
        "Q3-2025.pdf", 3,
    ),
    _page(
        "Self-attention computes a weighted mixture over the input sequence, "
        "while cross-attention gates on the encoder output.",
        "Transformer-Notes.pdf", 1,
    ),
]


@pytest.fixture()
def settings() -> Settings:
    return Settings(generation_mode="extractive", generation_top_pages=3)


def test_extractive_answers_from_text(settings: Settings) -> None:
    gen = ExtractiveGenerator(settings)
    out = gen.generate("What was the revenue in Q3?", PAGES)
    assert out.backend == "extractive"
    assert "$37M" in out.answer
    assert "Q3-2025.pdf (page 1)" in out.answer
    assert out.citations


def test_extractive_picks_relevant_page(settings: Settings) -> None:
    gen = ExtractiveGenerator(settings)
    out = gen.generate("What is the LFP battery cost per kWh?", PAGES)
    assert "$45" in out.answer
    assert "Q3-2025.pdf (page 3)" in out.answer


def test_extractive_no_match_is_honest(settings: Settings) -> None:
    gen = ExtractiveGenerator(settings)
    out = gen.generate("quantum computing entanglement", PAGES)
    assert "No directly matching text" in out.answer
    assert out.answer.strip()


def test_extractive_empty_query_tokens(settings: Settings) -> None:
    gen = ExtractiveGenerator(settings)
    out = gen.generate("the a an", PAGES)
    assert out.answer == ""
    assert out.note


def test_text_generator_returns_answer_untouched(settings: Settings) -> None:
    ok = Generation(answer="real llm answer", backend="text", model="m")
    class FakeText(TextGenerator):
        def generate(self, query, pages):
            return ok
    out = generate_with_fallback(FakeText(settings), "q", PAGES)
    assert out is ok
    assert out.answer == "real llm answer"


def test_fallback_on_empty_answer(settings: Settings) -> None:
    class Broken(TextGenerator):
        def generate(self, query, pages):
            return Generation(answer="", backend="text", note="generation backend unreachable: boom")
    out = generate_with_fallback(Broken(settings), "What was the revenue in Q3?", PAGES)
    assert out.backend == "extractive"
    assert "$37M" in out.answer
    assert "fallback: extractive" in out.note
    assert "unreachable" in out.note


def test_fallback_on_exception(settings: Settings) -> None:
    class Throwing(TextGenerator):
        def generate(self, query, pages):
            raise RuntimeError("boom")
    out = generate_with_fallback(Throwing(settings), "LFP battery cost?", PAGES)
    assert out.backend == "extractive"
    assert "$45" in out.answer
    assert "failed" in out.note


def test_extractive_mode_skips_fallback(settings: Settings) -> None:
    gen = ExtractiveGenerator(settings)
    out = generate_with_fallback(gen, "What was the revenue in Q3?", PAGES)
    assert out.backend == "extractive"
    assert out.note == ""  # direct call, no fallback note


def test_get_generator_routing() -> None:
    assert isinstance(get_generator(Settings(generation_mode="extractive")), ExtractiveGenerator)
    assert isinstance(get_generator(Settings(generation_mode="text")), TextGenerator)
    assert isinstance(get_generator(Settings(generation_mode="vision")), VisionGenerator)
    assert isinstance(get_generator(Settings()), TextGenerator)