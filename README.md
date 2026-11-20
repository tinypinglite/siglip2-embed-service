# SakuraMedia SigLIP2 Embedding Service

[简体中文](./README.zh-CN.md) | English

Provides SigLIP2 image and text embeddings for SakuraMedia image search. The image bundles pre-exported models and runtimes: it does not download models at startup, mount a model directory, or include PyTorch.

The service is fixed to the `siglip2-base-patch16-224-v3` embedding space and returns L2-normalized 768-dimensional vectors. Do not try to replace the model files. After changing the model or embedding space, existing image search indexes in SakuraMedia cannot be mixed with the new embeddings.

The image endpoint accepts static WebP and JPEG (`.jpg`/`.jpeg`) and detects the format from file contents rather than the uploaded filename. By default, the service accepts source images up to 40 megapixels, covering common 8K landscape images. This is an admission limit that prevents abnormal images from exhausting memory; it is not the model input size. libwebp outputs the 224×224 RGB data required by the model directly during decoding. JPEG is first downscaled by libjpeg-turbo during DCT decoding to at most 1 million RGB pixels, then resized to 224×224, so the full-size image is never expanded in memory.

Progressive JPEG requires the decoder to retain coefficients for the entire image, so it cannot be fully streamed row by row like baseline JPEG. To keep memory bounded, the service supports progressive JPEGs up to 10 megapixels with 32 scans. This additional limit applies only to progressive JPEGs and does not affect baseline JPEG, WebP, or the 40-megapixel `MAX_IMAGE_PIXELS` admission limit. Image CPU preprocessing concurrency is controlled by `CPU_CONCURRENCY` and defaults to 1. The general source-image admission limit can be adjusted with `MAX_IMAGE_PIXELS`.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `CPU_CONCURRENCY` | `1` | Number of concurrent image CPU preprocessing workers. |
| `TEXT_TOWER_ENABLED` | `true` | Whether the text tower is enabled. When set to `false`, the text encoder and tokenizer are not loaded and only image embeddings are provided; `POST /v1/embed/texts` returns 503, the dimension probe in `GET /v1/embedding-space` switches to images, and `modalities` still reports `image` and `text` for compatibility with existing clients. |
| `MAX_IMAGE_PIXELS` | `40000000` | Source image pixel admission limit. |
| `MAX_IMAGE_BYTES` | `20000000` | Per-image byte limit. |
| `MAX_REQUEST_BYTES` | `128000000` | Total request byte limit; must not be less than `MAX_IMAGE_BYTES`. |

At startup, the service automatically selects a backend based on the inference runtime actually installed in the image: with onnxruntime, it uses CUDA when available or CPU; with OpenVINO, it uses an Intel integrated GPU. No backend configuration is needed. `EMBEDDING_BACKEND` from older versions is deprecated and ignored if set.

## Image variants

| Image | Platform | Purpose |
| --- | --- | --- |
| `tinyping/siglip2-embed-service:cpu` | Linux amd64, arm64 | CPU only. Suitable for AMD CPUs, Intel CPUs, and ARM64 devices. |
| `tinyping/siglip2-embed-service:intel` | Linux amd64 | Intel integrated GPU. Uses OpenVINO and requires `/dev/dri` to be passed into the container. Supports Gen9 and newer Intel integrated GPUs (HD 500/600, UHD 600/605, UHD, Iris Xe, Arc, etc., covering Apollo Lake, Gemini Lake, Jasper Lake, Alder Lake-N, and Twin Lake). |
| `tinyping/siglip2-embed-service:cuda` | Linux amd64 | NVIDIA GPUs. Uses CUDA and requires the NVIDIA Container Toolkit. |

The `intel` image does not fall back to CPU: if the integrated GPU or driver is unsupported (for example, HD 400 (Braswell) and older integrated GPUs, AMD integrated GPUs, or `/dev/dri` not being passed in), the service fails to start and explains why in the logs; use the `cpu` image instead.

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](./LICENSE); non-commercial use only.

Copyright (C) 2026 tinypinglite
