"""Central configuration: .env (secrets/env) + config.yaml (tunables) + CLI overrides.

Resolution order: defaults < config.yaml < .env < CLI overrides.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

# ---- HF cache must be set at import time ----------------------------------
# huggingface_hub/transformers snapshot cache paths when they are first
# imported, so env vars set later (e.g. in configure_hf_env) are ignored.
# Set them here, before any HF import can happen. (setdefault so an explicit
# HF_HOME from the environment still wins.)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "data" / "hf-cache" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "data" / "hf-cache" / "transformers"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Settings(BaseSettings):
    """Runtime configuration. Env vars (prefixed as-is) override config.yaml + defaults."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- paths ----
    base_dir: Path = ROOT
    corpus_dir: Path = ROOT / "data" / "corpus"
    pages_dir: Path = ROOT / "data" / "pages"
    qdrant_path: Path = ROOT / "data" / "qdrant"
    hf_cache: Path = ROOT / "data" / "hf-cache"
    manifests_dir: Path = ROOT / "data" / "manifests"

    # ---- models ----
    colpali_model: str = "vidore/colqwen2-v1.0-hf"
    text_embed_model: str = "BAAI/bge-large-en-v1.5"
    text_sparse_model: str = "Qdrant/bm25"
    device: str = "auto"
    embed_batch_size: int = 2

    # ---- vector db (local embedded or hosted Qdrant) ----
    collection: str = "docs"
    colpali_dim: int = 128
    dense_dim: int = 1024
    distance: str = "Dot"
    # Hosted Qdrant cluster (recommended for a real deployment). Provide
    # QDRANT_URL + QDRANT_API_KEY in .env; empty qdrant_url => local embedded
    # mode (single writer = the API server process). These two are deliberately
    # NOT in config.yaml: yaml values become constructor args and would shadow
    # the .env credentials (constructor kwargs outrank env vars in
    # pydantic-settings).
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_timeout: float = 60.0  # per-request timeout (seconds) for cloud

    # ---- retrieval ----
    top_k: int = 8
    colpali_prefetch: int = 30
    dense_prefetch: int = 60
    sparse_prefetch: int = 60
    rrf_k: int = 60

    # ---- generation (OmniRoute gateway, free kilo models) ----
    omniroute_base_url: str = "http://localhost:8080/v1"
    omniroute_api_key: str = "local"
    omniroute_model: str = "kilo"
    generation_mode: str = "text"  # text | extractive | vision
    generation_top_pages: int = 3
    excerpt_chars: int = 16000
    temperature: float = 0.2
    max_tokens: int = 700
    llm_timeout: float = 90.0
    generation_retries: int = 2       # extra attempts when the gateway errors/returns empty
    generation_retry_delay: float = 2.0  # seconds between retries

    # ---- API ----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: str = ""  # empty = auth disabled

    # ---- runtime ----
    warmup_on_start: bool = False  # load embedders at boot (demo boxes: be ready instantly)

    # ---- OCR ----
    ocr_enabled: bool = False
    min_text_chars: int = 40

    def __init__(self, **kwargs: Any) -> None:
        yaml_cfg = _load_yaml(ROOT / "config.yaml")
        flat: dict[str, Any] = {}
        for section in yaml_cfg.values():
            if isinstance(section, dict):
                flat.update(section)
        flat.pop("paths", None)
        # paths from config.yaml are relative to repo root
        paths = yaml_cfg.get("paths", {})
        overrides: dict[str, Any] = {}
        for key, val in paths.items():
            overrides[key] = ROOT / val
        overrides.update(flat)
        overrides.update(kwargs)
        super().__init__(**overrides)

    # ---- resolved helpers ----
    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # noqa: F401

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def ensure_dirs(self) -> None:
        for d in (self.corpus_dir, self.pages_dir, self.manifests_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.qdrant_path and not self.is_cloud:
            self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.hf_cache.mkdir(parents=True, exist_ok=True)

    @property
    def is_cloud(self) -> bool:
        """True when running against a hosted Qdrant cluster."""
        return bool(self.qdrant_url)

    def summary(self) -> dict[str, Any]:
        return {
            "colpali_model": self.colpali_model,
            "text_embed_model": self.text_embed_model,
            "device": self.resolved_device,
            "collection": self.collection,
            "qdrant": "cloud" if self.is_cloud else "local",
            "generation_mode": self.generation_mode,
            "generation_model": self.omniroute_model,
            "generation_base_url": self.omniroute_base_url,
            "ocr_enabled": self.ocr_enabled,
            "api": f"{self.api_host}:{self.api_port}",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def mlflow_tracking_uri(settings: Settings) -> str:
    """SQLite-backed MLflow tracking URI under data/mlruns.

    MLflow >= 3 refuses the legacy file-store backend, so we use sqlite.
    """
    db = settings.base_dir / "data" / "mlruns" / "mlflow.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + db.as_posix()


def configure_hf_env(settings: Settings) -> None:
    """Point HF/transformers caches at the project data dir (D: drive).

    huggingface_hub reads these at import time, so after the fact we also
    patch its already-imported constants to match the (possibly overridden)
    settings values.
    """
    os.environ["HF_HOME"] = str(settings.hf_cache)
    os.environ["HF_HUB_CACHE"] = str(settings.hf_cache / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(settings.hf_cache / "transformers")
    try:
        import huggingface_hub.constants as hfc

        hfc.HF_HOME = settings.hf_cache
        hfc.HF_HUB_CACHE = Path(settings.hf_cache) / "hub"
        hfc.TRANSFORMERS_CACHE = Path(settings.hf_cache) / "transformers"
    except Exception:
        pass  # older huggingface_hub: env vars above suffice for fresh imports