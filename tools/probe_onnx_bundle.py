"""Run the exported static-batch SigLIP2 ONNX towers without PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), required=True)
    return parser.parse_args()


def run_session(session: ort.InferenceSession, inputs: dict[str, np.ndarray]) -> list[int]:
    output = session.run(None, inputs)[0]
    if output.shape != (1, 768):
        raise RuntimeError(f"unexpected output shape: {output.shape}")
    if not np.isfinite(output).all():
        raise RuntimeError("model output contains non-finite values")
    return [int(value) for value in output.shape]


def main() -> None:
    args = parse_args()
    provider = "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"required provider is unavailable: {provider}")

    image_session = ort.InferenceSession(
        args.model_path / "image_encoder.onnx", providers=[provider]
    )
    text_session = ort.InferenceSession(
        args.model_path / "text_encoder.onnx", providers=[provider]
    )
    image_shape = [int(value) for value in image_session.get_inputs()[0].shape]
    text_length = int(text_session.get_inputs()[0].shape[1])
    image_shape = run_session(
        image_session,
        {"pixel_values": np.zeros(image_shape, dtype=np.float16)},
    )
    text_shape = run_session(
        text_session,
        {
            "input_ids": np.zeros((1, text_length), dtype=np.int64),
            "attention_mask": np.ones((1, text_length), dtype=np.int64),
        },
    )
    print(
        json.dumps(
            {
                "provider": provider,
                "image_shape": image_shape,
                "text_shape": text_shape,
            }
        )
    )


if __name__ == "__main__":
    main()
