"""PyTorch-free loading and bounded embedding execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
from transformers.image_utils import ChannelDimension

from .settings import Settings


class InputError(ValueError):
    """The caller supplied an image that cannot be processed safely."""


class Backend(Protocol):
    def embed_images(self, pixels: list[np.ndarray]) -> np.ndarray: ...

    def embed_texts(self, tokens: dict[str, np.ndarray]) -> np.ndarray: ...

    def prepare_image(self, image: Image.Image) -> np.ndarray: ...

    def prepare_texts(self, texts: list[str]) -> dict[str, np.ndarray]: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("model returned a zero embedding")
    return vectors / norms


class _BaseBackend:
    def __init__(self, model_path: Path) -> None:
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.text_length = int(config.text_config.max_position_embeddings)

    def prepare_image(self, image: Image.Image) -> np.ndarray:
        return np.asarray(
            self.image_processor(
                images=[image],
                return_tensors="np",
                input_data_format=ChannelDimension.LAST,
            )["pixel_values"]
        )

    def prepare_texts(self, texts: list[str]) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.text_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )
        input_ids = np.asarray(tokens["input_ids"])
        attention_mask = np.asarray(tokens["attention_mask"])
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise RuntimeError("SigLIP2 tokenizer must define an EOS token")

        # SigLIP2 pools the final position, which must be a sticky EOS token.
        input_ids = np.where(attention_mask == 0, eos_token_id, input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(attention_mask),
        }


class OnnxBackend(_BaseBackend):
    def __init__(self, settings: Settings, provider: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("ONNX Runtime is unavailable in this image") from error

        if provider not in ort.get_available_providers():
            raise RuntimeError(f"required ONNX Runtime provider is unavailable: {provider}")

        super().__init__(settings.model_path)
        self.provider = provider
        self.image_session = self._load_session(
            ort, settings.model_path / "image_encoder.onnx"
        )
        self.text_session = self._load_session(
            ort, settings.model_path / "text_encoder.onnx"
        )
        input_type = self.image_session.get_inputs()[0].type
        self.image_dtype = np.float16 if input_type == "tensor(float16)" else np.float32

    def _load_session(self, ort, model_path: Path):
        if not model_path.is_file():
            raise RuntimeError(f"missing bundled model: {model_path}")
        session = ort.InferenceSession(str(model_path), providers=[self.provider])
        if session.get_providers()[0] != self.provider:
            raise RuntimeError(
                f"model did not initialize with {self.provider}: {session.get_providers()}"
            )
        return session

    def embed_images(self, pixels: list[np.ndarray]) -> np.ndarray:
        vectors = [
            self.image_session.run(
                None, {"pixel_values": pixel.astype(self.image_dtype, copy=False)}
            )[0]
            for pixel in pixels
        ]
        return _normalize(np.concatenate(vectors, axis=0))

    def embed_texts(self, tokens: dict[str, np.ndarray]) -> np.ndarray:
        vectors = [
            self.text_session.run(
                None,
                {
                    "input_ids": input_ids[None, :],
                    "attention_mask": attention_mask[None, :],
                },
            )[0]
            for input_ids, attention_mask in zip(
                tokens["input_ids"], tokens["attention_mask"]
            )
        ]
        return _normalize(np.concatenate(vectors, axis=0))


class OpenVinoBackend(_BaseBackend):
    def __init__(self, settings: Settings) -> None:
        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError("OpenVINO is unavailable in this image") from error

        super().__init__(settings.model_path)
        image_path = settings.model_path / "image_encoder.xml"
        text_path = settings.model_path / "text_encoder.xml"
        if not image_path.is_file() or not text_path.is_file():
            raise RuntimeError("missing bundled OpenVINO model")

        core = ov.Core()
        self.image_compiled = core.compile_model(str(image_path), "GPU")
        self.text_compiled = core.compile_model(str(text_path), "GPU")
        self.image_request = self.image_compiled.create_infer_request()
        self.text_request = self.text_compiled.create_infer_request()
        self.image_input = self.image_compiled.input(0)
        self.text_input_ids = self.text_compiled.input(0)
        self.text_attention_mask = self.text_compiled.input(1)

    def embed_images(self, pixels: list[np.ndarray]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for pixel in pixels:
            self.image_request.infer({self.image_input: pixel})
            vectors.append(
                np.array(self.image_request.get_output_tensor(0).data, copy=True)
            )
        return _normalize(np.concatenate(vectors, axis=0))

    def embed_texts(self, tokens: dict[str, np.ndarray]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for input_ids, attention_mask in zip(
            tokens["input_ids"], tokens["attention_mask"]
        ):
            self.text_request.infer(
                {
                    self.text_input_ids: input_ids[None, :],
                    self.text_attention_mask: attention_mask[None, :],
                }
            )
            vectors.append(
                np.array(self.text_request.get_output_tensor(0).data, copy=True)
            )
        return _normalize(np.concatenate(vectors, axis=0))


def load_backend(settings: Settings) -> Backend:
    if not settings.model_path.is_dir():
        raise RuntimeError(f"bundled model directory does not exist: {settings.model_path}")
    if settings.backend == "intel_gpu":
        return OpenVinoBackend(settings)
    provider = (
        "CUDAExecutionProvider"
        if settings.backend == "cuda"
        else "CPUExecutionProvider"
    )
    return OnnxBackend(settings, provider)


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend = load_backend(settings)
        self._cpu_pool = ThreadPoolExecutor(max_workers=settings.cpu_concurrency)
        self._processor_lock = Lock()
        self._inference_lock = Lock()

    def embed_images(self, payloads: list[bytes]) -> np.ndarray:
        prepared = list(self._cpu_pool.map(self._prepare_one_image, payloads))
        with self._inference_lock:
            return self.backend.embed_images(prepared)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        with self._processor_lock:
            tokens = self.backend.prepare_texts(texts)
        with self._inference_lock:
            return self.backend.embed_texts(tokens)

    def _prepare_one_image(self, payload: bytes) -> np.ndarray:
        image = _decode_image(payload, self.settings.max_image_pixels)
        try:
            with self._processor_lock:
                return self.backend.prepare_image(image)
        finally:
            image.close()


def _decode_image(payload: bytes, max_pixels: int) -> Image.Image:
    try:
        with Image.open(BytesIO(payload)) as source:
            width, height = source.size
            if width * height > max_pixels:
                raise InputError(f"image exceeds the {max_pixels}-pixel limit")
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise InputError("invalid or unreadable image") from error
