from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(image_path):
    image = Image.open(Path(image_path)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def resize_rgb_image(image, max_dim=None):
    if max_dim is None:
        return image.astype(np.float64)

    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))

    if scale >= 1.0:
        return image.astype(np.float64)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = Image.fromarray(image).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float64)


def rgb_to_grayscale(rgb_image):
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected an RGB image of shape (m, n, 3).")

    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]

    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64)


def clip_to_uint8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_image(arr, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    arr_uint8 = clip_to_uint8(arr)

    if arr_uint8.ndim == 2:
        Image.fromarray(arr_uint8).save(save_path)
    elif arr_uint8.ndim == 3 and arr_uint8.shape[2] == 3:
        Image.fromarray(arr_uint8).save(save_path)
    else:
        raise ValueError(f"Unsupported image shape: {arr_uint8.shape}")


def compute_thin_svd(matrix):
    if matrix.ndim != 2:
        raise ValueError("SVD expects a 2D array.")
    return np.linalg.svd(matrix, full_matrices=False)


def compute_singular_values_only(matrix):
    if matrix.ndim != 2:
        raise ValueError("SVD expects a 2D array.")
    return np.linalg.svd(matrix, full_matrices=False, compute_uv=False)


def reconstruct_from_svd(U, s, Vt, k):
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1.")
    k = min(k, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def frobenius_error(A, A_k):
    diff = A.astype(np.float64) - A_k.astype(np.float64)
    return float(np.sqrt(np.sum(diff ** 2)))


def relative_frobenius_error(A, A_k):
    denom = np.sqrt(np.sum(A.astype(np.float64) ** 2))
    if denom == 0:
        return 0.0
    return float(frobenius_error(A, A_k) / denom)


def storage_ratio(m, n, k):
    return float(k * (m + n + 1) / (m * n))


def retained_energy(s, k):
    k = min(int(k), len(s))
    total = np.sum(s ** 2)
    if total == 0:
        return 0.0
    return float(np.sum(s[:k] ** 2) / total)


def ensure_valid_k_values(image_shape, k_values):
    max_rank = min(image_shape)
    valid = sorted({int(k) for k in k_values if 1 <= int(k) <= max_rank})

    if not valid:
        raise ValueError(f"No valid k values in [1, {max_rank}].")

    return valid


def eckart_young_error(s, k):
    k = min(int(k), len(s))
    return float(np.sqrt(np.sum(s[k:] ** 2)))
