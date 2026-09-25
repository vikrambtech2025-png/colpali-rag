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
    _is_grounded,
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
    # default mode comes from .env/config.yaml (env-dependent); whatever it resolves
    # to, routing must map it to a real backend
    assert isinstance(get_generator(Settings()), (ExtractiveGenerator, TextGenerator, VisionGenerator))


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


# ---- VisionGenerator (stubbed openai client, real page-image files) ----

def _stub_vision_openai(monkeypatch) -> list[dict]:
    calls: list[tuple[dict, str]] = []
    captured: list[dict] = []

    class FakeCompletions:
        @staticmethod
        def create(*args, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="The chart shows 2.5 dB link margin at 540 km."
            ))])

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", FakeClient)
    return captured


def _vision_pages(pages_dir, *images: str) -> list[PageResult]:
    from pathlib import Path

    pages: list[PageResult] = []
    for i, img in enumerate(images):
        path = Path(img)
        file = pages_dir / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")
        page = i + 1
        pages.append(PageResult(
            score=0.9 - i * 0.1, src="rpt.pdf", page=page,
            text="", image=f"rpt.pdf/page-{page}.png",
        ))
    return pages


def test_vision_generator_sends_page_images_as_data_uris(monkeypatch, tmp_path) -> None:
    captured = _stub_vision_openai(monkeypatch)
    pages_dir = tmp_path / "pages"
    pages = _vision_pages(pages_dir, "rpt.pdf/page-1.png", "rpt.pdf/page-2.png")
    settings = Settings(
        generation_mode="vision", pages_dir=pages_dir, generation_top_pages=2,
        omniroute_vision_model="vision-test", llm_timeout=5,
        generation_retries=0, generation_retry_delay=0.0,
    )
    out = VisionGenerator(settings).generate("What does the chart show?", pages)
    assert out.backend == "vision"
    assert out.model == "vision-test"
    assert "540 km" in out.answer
    assert out.citations
    assert len(captured) == 1
    assert captured[0]["model"] == "vision-test"
    content = captured[0]["messages"][1]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    texts = [c for c in content if c["type"] == "text"]
    assert len(images) == 2  # one data URI per retrieved page image
    assert all(c["image_url"]["url"].startswith("data:image/png;base64,") for c in images)
    assert "QUESTION:" in texts[0]["text"]


def test_vision_generator_skips_missing_images(monkeypatch, tmp_path) -> None:
    captured = _stub_vision_openai(monkeypatch)
    pages_dir = tmp_path / "pages"
    pages = _vision_pages(pages_dir, "rpt.pdf/page-1.png")  # page-2 file intentionally absent
    pages.append(PageResult(score=0.7, src="rpt.pdf", page=2, text="", image="rpt.pdf/page-2.png"))
    settings = Settings(
        generation_mode="vision", pages_dir=pages_dir, generation_top_pages=3,
        omniroute_vision_model="vision-test", generation_retries=0,
    )
    out = VisionGenerator(settings).generate("chart?", pages)
    assert out.backend == "vision"
    content = captured[0]["messages"][1]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 1  # missing file skipped, still answers from what exists


def test_vision_generator_no_images_is_honest(tmp_path) -> None:
    settings = Settings(generation_mode="vision", pages_dir=tmp_path / "pages", generation_top_pages=2)
    pages = [PageResult(score=0.9, src="rpt.pdf", page=1, text="hello world")]
    out = VisionGenerator(settings).generate("what?", pages)
    assert out.answer == ""
    assert "no page images" in out.note


# ---- grounded-answer guard (RAG Doctor: never ship invented citations) ----

def test_ungrounded_refusal_falls_back(monkeypatch, settings) -> None:
    _stub_openai(
        monkeypatch,
        ["I don't see any attached page images in this conversation. Please provide the document."],
    )
    out = generate_with_fallback(TextGenerator(settings), "Revenue?", PAGES)
    assert out.backend == "extractive"
    assert "ungrounded" in out.note
    assert "fallback" in out.note
    assert "$37M" in out.answer


def test_hallucinated_citation_falls_back(monkeypatch, settings) -> None:
    # cites page 9 of a 3-page list -> page was never retrieved
    _stub_openai(monkeypatch, ["Revenue was $99B [Q3-2025.pdf (page 9)]."])
    out = generate_with_fallback(TextGenerator(settings), "Revenue?", PAGES)
    assert out.backend == "extractive"
    assert "ungrounded" in out.note
    assert "$37M" in out.answer


def test_grounded_gateway_answer_passes(monkeypatch, settings) -> None:
    _stub_openai(monkeypatch, ["Revenue grew to $37M in Q3 2025 [Q3-2025.pdf (page 1)]."])
    out = generate_with_fallback(TextGenerator(settings), "Revenue?", PAGES)
    assert out.backend == "text"
    assert out.note == ""
    assert "$37M" in out.answer


def test_hallucinated_other_doc_falls_back(monkeypatch, settings) -> None:
    # cites a document that was never retrieved at all
    _stub_openai(monkeypatch, ["Revenue grew [Phantom-Doc.pdf (page 2)]."])
    out = generate_with_fallback(TextGenerator(settings), "Revenue?", PAGES)
    assert out.backend == "extractive"
    assert "ungrounded" in out.note


def test_is_grounded_helpers() -> None:
    assert _is_grounded("Revenue grew [Q3-2025.pdf (page 1)].", PAGES)
    assert not _is_grounded("Revenue grew [Q3-2025.pdf (page 9)].", PAGES)
    assert not _is_grounded("Revenue grew [Q3-2025.pdf (page 2)].", PAGES)  # page exists but not retrieved
    assert not _is_grounded("Revenue grew [Phantom-Doc.pdf (page 1)].", PAGES)
    assert not _is_grounded("I don't see any page images.", PAGES)
    assert not _is_grounded("Please provide the document and I will answer.", PAGES)
    assert _is_grounded("Plain answer with no citations.", PAGES)