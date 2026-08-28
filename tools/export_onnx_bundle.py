"""Export the two SigLIP2 towers once for a PyTorch-free runtime image."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer


class ImageTower(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values=pixel_values)


class TextTower(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    image_size = processor.size
    height = int(image_size.get("height") or image_size["shortest_edge"])
    width = int(image_size.get("width") or image_size["shortest_edge"])
    text_length = int(config.text_config.max_position_embeddings)

    model = AutoModel.from_pretrained(args.model_path, local_files_only=True).eval()
    if args.precision == "fp16":
        model = model.half()
    model = model.to(args.device)
    tensor_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    image_example = torch.zeros(
        (1, 3, height, width), dtype=tensor_dtype, device=args.device
    )
    text_ids = torch.zeros((1, text_length), dtype=torch.long, device=args.device)
    text_mask = torch.ones((1, text_length), dtype=torch.long, device=args.device)

    torch.onnx.export(
        ImageTower(model),
        (image_example,),
        args.output_dir / "image_encoder.onnx",
        input_names=("pixel_values",),
        output_names=("embeddings",),
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        TextTower(model),
        (text_ids, text_mask),
        args.output_dir / "text_encoder.onnx",
        input_names=("input_ids", "attention_mask"),
        output_names=("embeddings",),
        opset_version=17,
        dynamo=False,
    )

    processor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    shutil.copy2(args.model_path / "config.json", args.output_dir / "config.json")


if __name__ == "__main__":
    main()
