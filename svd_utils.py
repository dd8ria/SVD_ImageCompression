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


def _normalize(vec, tol=1e-12):
    norm = np.linalg.norm(vec)
    if norm < tol:
        return None
    return vec / norm


def _orthogonalize_against_basis(vec, basis, tol=1e-12):
    out = vec.astype(np.float64, copy=True)
    for b in basis:
        out -= np.dot(b, out) * b
    return _normalize(out, tol=tol)


def _power_iteration_symmetric(matrix, basis=None, max_iter=2000, tol=1e-10, seed=42):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Power iteration expects a square matrix.")

    n = matrix.shape[0]
    basis = [] if basis is None else basis
    rng = np.random.default_rng(seed)

    vector = None
    for _ in range(10):
        candidate = rng.standard_normal(n)
        candidate = _orthogonalize_against_basis(candidate, basis, tol=tol)
        if candidate is not None:
            vector = candidate
            break
    if vector is None:
        return 0.0, np.zeros(n, dtype=np.float64)

    for _ in range(max_iter):
        next_vector = matrix @ vector
        next_vector = _orthogonalize_against_basis(next_vector, basis, tol=tol)

        if next_vector is None:
            eigenvalue = float(vector @ (matrix @ vector))
            return max(eigenvalue, 0.0), vector

        if min(
            np.linalg.norm(next_vector - vector),
            np.linalg.norm(next_vector + vector),
        ) < tol:
            vector = next_vector
            break

        vector = next_vector

    eigenvalue = float(vector @ (matrix @ vector))
    return max(eigenvalue, 0.0), vector


def _manual_symmetric_eigendecomposition(matrix, tol=1e-10, max_iter=2000, seed=42):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Eigen-decomposition expects a square matrix.")

    n = matrix.shape[0]
    eigenvalues = []
    eigenvectors = []

    for idx in range(n):
        eigenvalue, eigenvector = _power_iteration_symmetric(
            matrix,
            basis=eigenvectors,
            max_iter=max_iter,
            tol=tol,
            seed=seed + idx,
        )

        if eigenvalue <= tol or np.linalg.norm(eigenvector) < tol:
            break

        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

    if not eigenvalues:
        return np.array([], dtype=np.float64), np.zeros((n, 0), dtype=np.float64)

    eigenvalues = np.array(eigenvalues, dtype=np.float64)
    eigenvectors = np.column_stack(eigenvectors)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors


def compute_thin_svd(matrix, tol=1e-10, max_iter=2000, seed=42):
    if matrix.ndim != 2:
        raise ValueError("SVD expects a 2D array.")

    A = matrix.astype(np.float64)
    m, n = A.shape

    if m >= n:
        gram = A.T @ A
        eigenvalues, V = _manual_symmetric_eigendecomposition(
            gram, tol=tol, max_iter=max_iter, seed=seed
        )
        singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))

        if singular_values.size == 0:
            return np.zeros((m, 0)), np.array([], dtype=np.float64), np.zeros((0, n))

        keep = singular_values > tol
        singular_values = singular_values[keep]
        V = V[:, keep]

        U_cols = []
        kept_s = []

        for idx, sigma in enumerate(singular_values):
            u = (A @ V[:, idx]) / sigma
            u = _normalize(u, tol=tol)
            if u is None:
                continue
            U_cols.append(u)
            kept_s.append(sigma)

        if not U_cols:
            return np.zeros((m, 0)), np.array([], dtype=np.float64), np.zeros((0, n))

        U = np.column_stack(U_cols)
        singular_values = np.array(kept_s, dtype=np.float64)
        r = len(singular_values)
        V = V[:, :r]
        Vt = V.T
        return U, singular_values, Vt

    gram = A @ A.T
    eigenvalues, U = _manual_symmetric_eigendecomposition(
        gram, tol=tol, max_iter=max_iter, seed=seed
    )
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))

    if singular_values.size == 0:
        return np.zeros((m, 0)), np.array([], dtype=np.float64), np.zeros((0, n))

    keep = singular_values > tol
    singular_values = singular_values[keep]
    U = U[:, keep]

    V_cols = []
    kept_s = []

    for idx, sigma in enumerate(singular_values):
        v = (A.T @ U[:, idx]) / sigma
        v = _normalize(v, tol=tol)
        if v is None:
            continue
        V_cols.append(v)
        kept_s.append(sigma)

    if not V_cols:
        return np.zeros((m, 0)), np.array([], dtype=np.float64), np.zeros((0, n))

    V = np.column_stack(V_cols)
    singular_values = np.array(kept_s, dtype=np.float64)
    r = len(singular_values)
    U = U[:, :r]
    Vt = V.T
    return U, singular_values, Vt


def compute_singular_values_only(matrix, tol=1e-10, max_iter=2000, seed=42):
    _, s, _ = compute_thin_svd(matrix, tol=tol, max_iter=max_iter, seed=seed)
    return s


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