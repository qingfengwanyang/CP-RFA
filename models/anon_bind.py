#!/usr/bin/env python3
"""Canonical digest of aligned anonymous face crops for token binding."""

import hashlib

import cv2
import numpy as np

BIND_DIGEST_LEN = 32  # SHA-256
DEFAULT_CROP_SIZE = 224


def canonicalize_anon_bgr(bgr: np.ndarray, crop_size: int = DEFAULT_CROP_SIZE) -> np.ndarray:
    """Fixed-size RGB uint8 array for deterministic hashing."""
    if bgr is None or bgr.size == 0:
        raise ValueError('empty anon crop')
    h, w = bgr.shape[:2]
    if h != crop_size or w != crop_size:
        bgr = cv2.resize(bgr, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def anon_crop_digest(bgr: np.ndarray, crop_size: int = DEFAULT_CROP_SIZE) -> bytes:
    """SHA-256 over canonical RGB pixels of the anonymous crop."""
    arr = canonicalize_anon_bgr(bgr, crop_size)
    return hashlib.sha256(arr.tobytes()).digest()


def verify_anon_bind(bgr: np.ndarray, expected_digest: bytes,
                     crop_size: int = DEFAULT_CROP_SIZE) -> bool:
    if expected_digest is None or len(expected_digest) != BIND_DIGEST_LEN:
        return False
    return anon_crop_digest(bgr, crop_size) == expected_digest
