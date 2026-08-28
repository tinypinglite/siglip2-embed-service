from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Barrier, Lock
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from transformers.image_utils import ChannelDimension

from siglip2_embed.runtime import (
    EmbeddingService,
    InputError,
    OnnxBackend,
    OpenVinoBackend,
    _decode_image,
)
from siglip2_embed.settings import EMBEDDING_SPACE_ID, MODEL_PATH, Settings


class _RecordingImageProcessor:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None
        self.pixel_values = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def __call__(self, **kwargs: object) -> dict[str, np.ndarray]:
        self.kwargs = kwargs
        return {"pixel_values": self.pixel_values}


class _RecordingTokenizer:
    eos_token_id = 1

    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, np.ndarray]:
        self.kwargs = kwargs
        return {
            "input_ids": np.array([[42, 1, 0, 0], [43, 44, 1, 0]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.int64),
        }


class _ConcurrentImageBackend:
    def __init__(self) -> None:
        self.barrier = Barrier(2)

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        self.barrier.wait(timeout=2)
        return np.zeros((1, 3, 224, 224), dtype=np.float32)

    def embed_images(self, pixels: list[np.ndarray]) -> np.ndarray:
        return np.zeros((len(pixels), 1), dtype=np.float32)


@pytest.mark.parametrize("backend_type", [OnnxBackend, OpenVinoBackend])
def test_prepare_image_declares_channels_last_for_tiny_rgb_image(backend_type: type) -> None:
    payload = BytesIO()
    Image.new("RGB", (1, 1), (0, 0, 0)).save(payload, format="WEBP")
    image = _decode_image(payload.getvalue(), max_pixels=16_000_000)
    processor = _RecordingImageProcessor()
    backend = object.__new__(backend_type)
    backend.image_processor = processor

    np.testing.assert_array_equal(backend.prepare_image(image), processor.pixel_values)
    assert processor.kwargs == {
        "images": [image],
        "do_resize": False,
        "return_tensors": "np",
        "input_data_format": ChannelDimension.LAST,
    }


@pytest.mark.parametrize("backend_type", [OnnxBackend, OpenVinoBackend])
def test_prepare_texts_uses_sticky_eos(backend_type: type) -> None:
    tokenizer = _RecordingTokenizer()
    backend = object.__new__(backend_type)
    backend.tokenizer = tokenizer
    backend.text_length = 64

    result = backend.prepare_texts(["first", "second"])

    assert tokenizer.kwargs == {
        "padding": "max_length",
        "max_length": 64,
        "truncation": True,
        "return_attention_mask": True,
        "return_tensors": "np",
    }
    np.testing.assert_array_equal(
        result["input_ids"],
        np.array([[42, 1, 1, 1], [43, 44, 1, 1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(result["attention_mask"], np.ones((2, 4), dtype=np.int64))


def test_settings_fix_the_model_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "/unexpected")
    monkeypatch.setenv("EMBEDDING_SPACE_ID", "unexpected")
    settings = Settings.from_env()

    assert settings.model_path == MODEL_PATH
    assert settings.embedding_space_id == EMBEDDING_SPACE_ID


def test_settings_accept_common_8k_source_images_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAX_IMAGE_PIXELS", raising=False)

    settings = Settings.from_env()

    assert settings.max_image_pixels == 40_000_000
    assert settings.max_image_pixels >= 8192 * 4096


def test_decode_image_accepts_source_at_pixel_limit() -> None:
    payload = BytesIO()
    Image.new("RGB", (8, 5), (0, 0, 0)).save(payload, format="WEBP")

    image = _decode_image(payload.getvalue(), max_pixels=40)

    assert image.shape == (224, 224, 3)


def test_decode_image_rejects_source_above_pixel_limit() -> None:
    payload = BytesIO()
    Image.new("RGB", (8, 5), (0, 0, 0)).save(payload, format="WEBP")

    with pytest.raises(InputError, match="image exceeds the 39-pixel limit"):
        _decode_image(payload.getvalue(), max_pixels=39)


def test_decode_image_rejects_non_webp_input() -> None:
    payload = BytesIO()
    Image.new("RGB", (8, 5), (0, 0, 0)).save(payload, format="PNG")

    with pytest.raises(InputError, match="only WebP images are supported"):
        _decode_image(payload.getvalue(), max_pixels=40)


def test_decode_image_rejects_animated_webp() -> None:
    payload = BytesIO()
    frames = [Image.new("RGB", (8, 5), color) for color in ("red", "blue")]
    frames[0].save(
        payload,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=100,
    )

    with pytest.raises(InputError, match="animated WebP images are not supported"):
        _decode_image(payload.getvalue(), max_pixels=40)


@pytest.mark.parametrize("lossless", [False, True])
def test_decode_image_scales_wide_webp_to_model_input(lossless: bool) -> None:
    payload = BytesIO()
    Image.new("RGB", (800, 200), (12, 34, 56)).save(
        payload, format="WEBP", lossless=lossless
    )

    image = _decode_image(payload.getvalue(), max_pixels=160_000)

    assert image.shape == (224, 224, 3)
    np.testing.assert_allclose(image[112, 112], [12, 34, 56], atol=2)


def test_image_preparation_honors_cpu_concurrency() -> None:
    payload = BytesIO()
    Image.new("RGB", (8, 5), (0, 0, 0)).save(payload, format="WEBP")
    service = object.__new__(EmbeddingService)
    service.settings = SimpleNamespace(max_image_pixels=40)
    service.backend = _ConcurrentImageBackend()
    service._cpu_pool = ThreadPoolExecutor(max_workers=2)
    service._inference_lock = Lock()

    try:
        result = service.embed_images([payload.getvalue(), payload.getvalue()])
    finally:
        service._cpu_pool.shutdown()

    assert result.shape == (2, 1)
