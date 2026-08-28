from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from transformers.image_utils import ChannelDimension

from siglip2_embed.runtime import OnnxBackend, OpenVinoBackend, _decode_image
from siglip2_embed.settings import EMBEDDING_SPACE_ID, MODEL_PATH, Settings


class _RecordingImageProcessor:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None
        self.pixel_values = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def __call__(self, **kwargs: object) -> dict[str, np.ndarray]:
        self.kwargs = kwargs
        return {"pixel_values": self.pixel_values}


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


def test_settings_fix_the_model_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "/unexpected")
    monkeypatch.setenv("EMBEDDING_SPACE_ID", "unexpected")
    settings = Settings.from_env()

    assert settings.model_path == MODEL_PATH
    assert settings.embedding_space_id == EMBEDDING_SPACE_ID
