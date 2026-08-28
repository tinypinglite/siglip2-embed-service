"""Bounded native decoding for static WebP model inputs."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


class WebPDecodeError(ValueError):
    """The payload cannot be decoded as a supported WebP image."""


_library_path = Path(__file__).with_name("_webp_decode.so")
try:
    _library = ctypes.CDLL(str(_library_path))
except OSError as error:
    raise RuntimeError(f"native WebP decoder is unavailable: {_library_path}") from error

_library.sakura_webp_get_features.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_library.sakura_webp_get_features.restype = ctypes.c_int
_library.sakura_webp_decode_rgb_scaled.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
_library.sakura_webp_decode_rgb_scaled.restype = ctypes.c_int


def decode_webp_rgb(
    payload: bytes, max_pixels: int, width: int, height: int
) -> np.ndarray:
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WEBP":
        raise WebPDecodeError("only WebP images are supported")

    data = ctypes.c_char_p(payload)
    source_width = ctypes.c_int()
    source_height = ctypes.c_int()
    has_animation = ctypes.c_int()
    status = _library.sakura_webp_get_features(
        data,
        len(payload),
        ctypes.byref(source_width),
        ctypes.byref(source_height),
        ctypes.byref(has_animation),
    )
    if status != 0:
        raise WebPDecodeError("invalid or unreadable WebP image")
    if has_animation.value:
        raise WebPDecodeError("animated WebP images are not supported")
    if source_width.value * source_height.value > max_pixels:
        raise WebPDecodeError(f"image exceeds the {max_pixels}-pixel limit")

    pixels = np.empty((height, width, 3), dtype=np.uint8)
    status = _library.sakura_webp_decode_rgb_scaled(
        data,
        len(payload),
        width,
        height,
        pixels.ctypes.data,
        pixels.nbytes,
    )
    if status == -1:
        raise RuntimeError("native WebP decoder ABI is incompatible")
    if status != 0:
        raise WebPDecodeError("invalid or unreadable WebP image")
    return pixels
