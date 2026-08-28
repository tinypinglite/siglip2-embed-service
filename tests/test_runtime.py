from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from transformers.image_utils import ChannelDimension

from siglip2_embed.runtime import InputError, OnnxBackend, OpenVinoBackend, _decode_image
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


@pytest.mark.parametrize("backend_type", [OnnxBackend, OpenVinoBackend])
def test_prepare_image_declares_channels_last_for_tiny_rgb_image(backend_type: type) -> None:
    payload = BytesIO()
    Image.new("RGB", (1, 1), (0, 0, 0)).save(payload, format="PNG")
    image = _decode_image(payload.getvalue(), max_pixels=16_000_000)
    processor = _RecordingImageProcessor()
    backend = object.__new__(backend_type)
    backend.image_processor = processor

    np.testing.assert_array_equal(backend.prepare_image(image), processor.pixel_values)
    assert processor.kwargs == {
        "images": [image],
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
    Image.new("RGB", (8, 5), (0, 0, 0)).save(payload, format="PNG")

    image = _decode_image(payload.getvalue(), max_pixels=40)

    assert image.size == (8, 5)


def test_decode_image_rejects_source_above_pixel_limit() -> None:
    payload = BytesIO()
    Image.new("RGB", (8, 5), (0, 0, 0)).save(payload, format="PNG")

    with pytest.raises(InputError, match="image exceeds the 39-pixel limit"):
        _decode_image(payload.getvalue(), max_pixels=39)
