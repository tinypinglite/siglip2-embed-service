"""Bounded native decoding for JPEG model inputs."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from PIL import Image


# libjpeg-turbo has to retain progressive coefficients before it can render
# scanlines. Keep that unavoidable allocation bounded independently of the
# broader source-image admission limit used by baseline JPEG and WebP.
MAX_PROGRESSIVE_JPEG_PIXELS = 10_000_000
MAX_PROGRESSIVE_JPEG_SCANS = 32
MAX_JPEG_DECODE_PIXELS = 1_000_000

_STATUS_SCAN_LIMIT = 2
_STATUS_OUTPUT_LIMIT = 3


class JPEGDecodeError(ValueError):
    """The payload cannot be decoded as a supported JPEG image."""


_library_path = Path(__file__).with_name("_jpeg_decode.so")
try:
    _library = ctypes.CDLL(str(_library_path))
except OSError as error:
    raise RuntimeError(f"native JPEG decoder is unavailable: {_library_path}") from error

_library.sakura_jpeg_get_features.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_library.sakura_jpeg_get_features.restype = ctypes.c_int
_library.sakura_jpeg_decode_rgb_scaled.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
_library.sakura_jpeg_decode_rgb_scaled.restype = ctypes.c_int


def decode_jpeg_rgb(
    payload: bytes, max_pixels: int, width: int, height: int
) -> np.ndarray:
    if len(payload) < 2 or payload[:2] != b"\xff\xd8":
        raise JPEGDecodeError("only JPEG images are supported")

    data = ctypes.c_char_p(payload)
    source_width = ctypes.c_int()
    source_height = ctypes.c_int()
    decoded_width = ctypes.c_int()
    decoded_height = ctypes.c_int()
    is_progressive = ctypes.c_int()
    status = _library.sakura_jpeg_get_features(
        data,
        len(payload),
        MAX_PROGRESSIVE_JPEG_SCANS,
        MAX_JPEG_DECODE_PIXELS,
        ctypes.byref(source_width),
        ctypes.byref(source_height),
        ctypes.byref(decoded_width),
        ctypes.byref(decoded_height),
        ctypes.byref(is_progressive),
    )
    if status == _STATUS_SCAN_LIMIT:
        raise JPEGDecodeError(
            f"progressive JPEG exceeds the {MAX_PROGRESSIVE_JPEG_SCANS}-scan limit"
        )
    if status == _STATUS_OUTPUT_LIMIT:
        raise JPEGDecodeError("JPEG image cannot be safely scaled")
    if status != 0:
        raise JPEGDecodeError("invalid or unreadable JPEG image")

    source_pixels = source_width.value * source_height.value
    if source_pixels > max_pixels:
        raise JPEGDecodeError(f"image exceeds the {max_pixels}-pixel limit")
    if is_progressive.value and source_pixels > MAX_PROGRESSIVE_JPEG_PIXELS:
        raise JPEGDecodeError(
            "progressive JPEG exceeds the "
            f"{MAX_PROGRESSIVE_JPEG_PIXELS}-pixel memory safety limit"
        )

    pixels = np.empty(
        (decoded_height.value, decoded_width.value, 3), dtype=np.uint8
    )
    status = _library.sakura_jpeg_decode_rgb_scaled(
        data,
        len(payload),
        decoded_width.value,
        decoded_height.value,
        pixels.ctypes.data,
        pixels.nbytes,
    )
    if status != 0:
        raise JPEGDecodeError("invalid or unreadable JPEG image")

    image = Image.fromarray(pixels)
    return np.asarray(
        image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8
    )
