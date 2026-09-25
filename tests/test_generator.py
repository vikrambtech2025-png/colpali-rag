"""Unit tests for the generator layer (no models, no network)."""
from __future__ import annotations

from types import SimpleNamespace

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


# ---- TextGenerator retry behavior (stubbed openai client) ----

def _stub_openai(monkeypatch, contents: list[str]) -> list[int]:
    """Replace openai.OpenAI with a fake whose completions return the given contents."""
    calls: list[int] = []

    class FakeCompletions:
        @staticmethod
        def create(*args, **kwargs):
            calls.append(1)
            content = contents[min(len(calls) - 1, len(contents) - 1)]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", FakeClient)
    return calls


def test_text_generator_exhausts_retries_on_empty(monkeypatch) -> None:
    calls = _stub_openai(monkeypatch, ["", "", ""])
    settings = Settings(generation_mode="text", generation_retries=2, generation_retry_delay=0.01)
    out = TextGenerator(settings).generate("What was the revenue in Q3?", PAGES)
    assert len(calls) == 3  # initial + 2 retries
    assert out.answer == ""
    assert "unreachable" in out.note and "empty reply" in out.note


def test_text_generator_retry_recovers(monkeypatch) -> None:
    calls = _stub_openai(monkeypatch, ["", "Revenue grew to $37M in Q3 2025."])
    settings = Settings(generation_mode="text", generation_retries=2, generation_retry_delay=0.01)
    out = TextGenerator(settings).generate("What was the revenue in Q3?", PAGES)
    assert len(calls) == 2  # failed once, recovered on retry
    assert "$37M" in out.answer
    assert out.backend == "text"
    assert out.note == ""


def test_text_generator_no_retries_when_disabled(monkeypatch) -> None:
    calls = _stub_openai(monkeypatch, [""])
    settings = Settings(generation_mode="text", generation_retries=0)
    out = TextGenerator(settings).generate("What was the revenue in Q3?", PAGES)
    assert len(calls) == 1
    assert out.answer == ""


def test_retry_config_defaults() -> None:
    s = Settings()
    assert s.generation_retries == 2
    assert s.generation_retry_delay == 2.0
    assert s.generation_retries == s.generation_retries + 0  # sanity, field is int