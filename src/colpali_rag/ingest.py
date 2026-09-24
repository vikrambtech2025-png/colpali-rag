"""Ingestion: PDF -> rendered pages + text -> ColPali + BGE-M3 vectors -> Qdrant.

Runs inside the API process (single Qdrant writer). Exposes a CLI too.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import Settings, configure_hf_env
from .embeddings import ColPaliEmbedder, TextEmbedder
from .qdrant_store import QdrantStore
from .utils import now_iso, slugify, timed

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    src: str
    pages: int = 0
    colpali_points: int = 0
    text_points: int = 0
    ocr_pages: int = 0
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "pages": self.pages,
            "colpali_points": self.colpali_points,
            "text_points": self.text_points,
            "ocr_pages": self.ocr_pages,
            "duration_s": round(self.duration_s, 2),
            "errors": self.errors,
        }


def render_pdf_pages(pdf_path: Path, pages_dir: Path, scale: float = 1.5) -> list[tuple[Path, str]]:
    """Render every page to PNG and extract text. Returns [(png_path, text)]."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    stem = slugify(pdf_path.stem)
    out_dir = pages_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[tuple[Path, str]] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        png = out_dir / f"page-{i:03d}.png"
        pix.save(str(png))
        pages.append((png, text))
    doc.close()
    return pages


def _ocr_text(png_path: Path) -> str:
    try:
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()
        result, _ = engine(str(png_path))
        if not result:
            return ""
        return "\n".join(line[1] for line in result)
    except Exception as exc:  # pragma: no cover
        log.warning("OCR failed for %s: %s", png_path, exc)
        return ""


class IngestPipeline:
    def __init__(self, settings: Settings, store: QdrantStore, colpali: ColPaliEmbedder, text: TextEmbedder) -> None:
        self.settings = settings
        self.store = store
        self.colpali = colpali
        self.text = text

    def ingest_pdf(self, pdf_path: str | Path) -> IngestReport:
        from time import perf_counter

        configure_hf_env(self.settings)
        self.store.ensure_collection()
        pdf = Path(pdf_path)
        report = IngestReport(src=pdf.name)
        t0 = perf_counter()
        with timed(f"render {pdf.name}"):
            pages = render_pdf_pages(pdf, self.settings.pages_dir)
        if not pages:
            report.errors.append("no pages rendered")
            return report

        # text layer (embedded text, OCR fallback)
        texts: list[str] = []
        for png, text in pages:
            if len(text.strip()) < self.settings.min_text_chars and self.settings.ocr_enabled:
                text = _ocr_text(png)
                if text:
                    report.ocr_pages += 1
            texts.append(text or "")

        # embeddings
        with timed(f"colpali embed {len(pages)} pages"):
            colpali_vecs: list[np.ndarray] = self.colpali.embed_images(
                [Image.open(png).convert("RGB") for png, _ in pages]
            )
        text_embs: list[tuple[np.ndarray, dict[int, float]]] = []
        for t in texts:
            if t.strip():
                try:
                    text_embs.append(self.text.embed_documents([t])[0])
                except Exception as exc:  # pragma: no cover
                    log.warning("text embed failed: %s", exc)
                    text_embs.append((np.zeros(self.settings.dense_dim, dtype=np.float32), {}))
            else:
                text_embs.append((np.zeros(self.settings.dense_dim, dtype=np.float32), {}))

        # upsert
        with timed(f"upsert {len(pages)} points"):
            for (png, text), vec, (dense, sparse) in zip(pages, colpali_vecs, text_embs):
                self.store.upsert_page(
                    src=pdf.name,
                    page=int(png.stem.split("-")[-1]),
                    colpali_vecs=vec,
                    dense_vec=dense,
                    sparse=sparse,
                    payload={
                        "src": pdf.name,
                        "page": int(png.stem.split("-")[-1]),
                        "text": text,
                        "image": str(png.relative_to(self.settings.pages_dir)).replace("\\", "/"),
                        "ingested_at": now_iso(),
                    },
                )
                report.colpali_points += 1
                if text.strip():
                    report.text_points += 1

        report.pages = len(pages)
        report.duration_s = perf_counter() - t0

        # MLflow run per document
        try:
            import mlflow

            mlflow.set_tracking_uri("file:" + str(self.settings.base_dir / "data" / "mlruns"))
            with mlflow.start_run(run_name=f"ingest-{slugify(pdf.name)}"):
                mlflow.log_params({"source": pdf.name, "colpali_model": self.settings.colpali_model})
                mlflow.log_metrics(
                    {"pages": report.pages, "colpali_points": report.colpali_points, "text_points": report.text_points}
                )
        except Exception as exc:  # pragma: no cover
            log.warning("mlflow logging skipped: %s", exc)

        manifest = self.settings.manifests_dir / f"{slugify(pdf.name)}.json"
        manifest.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        log.info("Ingested %s: %d pages in %.1fs", pdf.name, report.pages, report.duration_s)
        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli_main(argv: list[str] | None = None) -> int:
    configure_hf_env(Settings())
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="colpali-ingest", description="Ingest PDFs into the ColPali RAG index")
    parser.add_argument("paths", nargs="+", help="PDF files or directories")
    parser.add_argument("--collection", help="override Qdrant collection name")
    parser.add_argument("--no-mlflow", action="store_true", help="disable mlflow logging")
    args = parser.parse_args(argv)

    settings = Settings(collection=args.collection) if args.collection else Settings()
    settings.ensure_dirs()
    store = QdrantStore(settings)
    colpali = ColPaliEmbedder(settings)
    text = TextEmbedder(settings)
    pipeline = IngestPipeline(settings, store, colpali, text)

    files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.pdf")))
        elif path.is_file():
            files.append(path)
    if not files:
        print("No PDFs found.")
        return 1

    reports = []
    for f in files:
        try:
            reports.append(pipeline.ingest_pdf(f).to_dict())
        except Exception as exc:  # pragma: no cover
            print(f"FAILED {f}: {exc}")
            reports.append({"src": f.name, "errors": [str(exc)]})
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())