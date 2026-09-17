# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

FROM python:3.11-slim-bookworm AS python-build
COPY --from=uv /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libturbojpeg0-dev \
        libwebp-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_NO_DEV=1

COPY pyproject.toml uv.lock README.md ./

FROM python-build AS cpu-build
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra cpu
COPY src ./src
COPY tools/build_jpeg_decoder.sh tools/build_webp_decoder.sh ./tools/
RUN sh tools/build_webp_decoder.sh \
    && sh tools/build_jpeg_decoder.sh \
    && uv sync --frozen --no-dev --extra cpu

FROM python:3.11-slim-bookworm AS cpu-runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libturbojpeg0 libwebp7 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
COPY --from=cpu-build /app/.venv /app/.venv
COPY --from=cpu-build /app/src /app/src
RUN python -c 'import importlib.util; assert importlib.util.find_spec("torch") is None'

FROM cpu-runtime AS cpu
ARG MODEL_BUNDLE=model-bundles/onnx-fp16
COPY ${MODEL_BUNDLE}/ /opt/siglip2/
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "siglip2_embed.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

FROM python:3.11-slim-trixie AS intel-python-build
COPY --from=uv /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libturbojpeg0-dev \
        libwebp-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_NO_DEV=1

COPY pyproject.toml uv.lock README.md ./

FROM intel-python-build AS intel-build
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra intel
COPY src ./src
COPY tools/build_jpeg_decoder.sh tools/build_webp_decoder.sh ./tools/
RUN sh tools/build_webp_decoder.sh \
    && sh tools/build_jpeg_decoder.sh \
    && uv sync --frozen --no-dev --extra intel

FROM python:3.11-slim-trixie AS intel-runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libdrm2 \
        libtbb12 \
        libturbojpeg0 \
        libwebp7 \
        ocl-icd-libopencl1 \
        wget \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q -P /tmp/intel-drivers \
        https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3/intel-igc-core-2_2.36.3+21719_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3/intel-igc-opencl-2_2.36.3+21719_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/intel-opencl-icd_26.22.38646.4-0_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-core_1.0.17537.24_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-opencl_1.0.17537.24_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/intel-opencl-icd-legacy1_24.35.30872.36_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/libigdgmm12_22.10.0_amd64.deb \
    && dpkg -i /tmp/intel-drivers/*.deb \
    && rm -rf /tmp/intel-drivers \
    && apt-get purge -y wget \
    && rm -rf /var/lib/apt/lists/* \
    && test -f /etc/OpenCL/vendors/intel.icd \
    && test -f /etc/OpenCL/vendors/intel_legacy1.icd
WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
COPY --from=intel-build /app/.venv /app/.venv
COPY --from=intel-build /app/src /app/src
RUN python -c 'import importlib.util; assert importlib.util.find_spec("torch") is None'

FROM intel-runtime AS intel
ARG MODEL_BUNDLE=model-bundles/openvino-fp16
COPY ${MODEL_BUNDLE}/ /opt/siglip2/
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "siglip2_embed.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cuda-build
COPY --from=uv /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libturbojpeg0-dev \
        libwebp-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_NO_DEV=1
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra cuda --python 3.11
COPY src ./src
COPY tools/build_jpeg_decoder.sh tools/build_webp_decoder.sh ./tools/
RUN sh tools/build_webp_decoder.sh \
    && sh tools/build_jpeg_decoder.sh \
    && uv sync --frozen --no-dev --extra cuda --python 3.11

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cuda-runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libturbojpeg libwebp7 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
COPY --from=cuda-build /root/.local/share/uv/python /root/.local/share/uv/python
COPY --from=cuda-build /app/.venv /app/.venv
COPY --from=cuda-build /app/src /app/src
RUN python -c 'import importlib.util; assert importlib.util.find_spec("torch") is None'

FROM cuda-runtime AS cuda
ARG MODEL_BUNDLE=model-bundles/onnx-fp16
COPY ${MODEL_BUNDLE}/ /opt/siglip2/
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "siglip2_embed.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
