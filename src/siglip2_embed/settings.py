"""Environment-backed service settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


MODEL_PATH = Path("/opt/siglip2")
EMBEDDING_SPACE_ID = "siglip2-base-patch16-224-webp-v2"
VALID_BACKENDS = {"cpu", "cuda", "intel_gpu"}


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True)
class Settings:
    model_path: Path
    embedding_space_id: str
    backend: str
    cpu_concurrency: int
    max_image_bytes: int
    max_request_bytes: int
    max_image_pixels: int

    @classmethod
    def from_env(cls) -> "Settings":
        backend = os.getenv("EMBEDDING_BACKEND", "cpu")
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"EMBEDDING_BACKEND must be one of {sorted(VALID_BACKENDS)}"
            )

        max_image_bytes = _positive_int("MAX_IMAGE_BYTES", 20_000_000)
        max_request_bytes = _positive_int("MAX_REQUEST_BYTES", 128_000_000)
        if max_request_bytes < max_image_bytes:
            raise ValueError("MAX_REQUEST_BYTES must be at least MAX_IMAGE_BYTES")

        return cls(
            model_path=MODEL_PATH,
            embedding_space_id=EMBEDDING_SPACE_ID,
            backend=backend,
            cpu_concurrency=_positive_int("CPU_CONCURRENCY", 1),
            max_image_bytes=max_image_bytes,
            max_request_bytes=max_request_bytes,
            # This is a source-image safety ceiling, not the model input size.
            # Static WebP inputs are decoded directly to the 224x224 model size.
            max_image_pixels=_positive_int("MAX_IMAGE_PIXELS", 40_000_000),
        )
