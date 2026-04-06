import numpy as np

from svd_utils import (
    clip_to_uint8,
    compute_thin_svd,
    eckart_young_error,
    frobenius_error,
    relative_frobenius_error,
    reconstruct_from_svd,
    retained_energy,
    storage_ratio,
)


def prepare_grayscale_svd(image):
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")

    A = image.astype(np.float64)
    U, s, Vt = compute_thin_svd(A)

    return {
        "A": A,
        "U": U,
        "s": s,
        "Vt": Vt,
    }


def prepare_rgb_svd(image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image of shape (m, n, 3).")

    channels = []

    for channel_idx in range(3):
        channel = image[:, :, channel_idx].astype(np.float64)
        U, s, Vt = compute_thin_svd(channel)

        channels.append(
            {
                "A": channel,
                "U": U,
                "s": s,
                "Vt": Vt,
            }
        )

    return {
        "channels": channels,
        "shape": image.shape,
    }


def _build_grayscale_result(A, U, s, Vt, k):
    k = min(int(k), len(s))
    approx = reconstruct_from_svd(U, s, Vt, k)

    return {
        "k": k,
        "approx_float": approx,
        "approx_uint8": clip_to_uint8(approx),
        "frobenius_error": frobenius_error(A, approx),
        "relative_frobenius_error": relative_frobenius_error(A, approx),
        "storage_ratio": storage_ratio(A.shape[0], A.shape[1], k),
        "retained_energy": retained_energy(s, k),
    }


def compress_grayscale_svd(image, k):
    prepared = prepare_grayscale_svd(image)
    return _build_grayscale_result(
        prepared["A"], prepared["U"], prepared["s"], prepared["Vt"], k
    )


def compress_grayscale_svd_precomputed(prepared, k):
    return _build_grayscale_result(
        prepared["A"], prepared["U"], prepared["s"], prepared["Vt"], k
    )


def compress_rgb_svd(image, k):
    prepared = prepare_rgb_svd(image)
    return compress_rgb_svd_precomputed(prepared, k)


def compress_rgb_svd_precomputed(prepared, k):
    channels = prepared["channels"]
    m, n = prepared["shape"][:2]

    approx_channels = []
    total_energy = 0.0
    kept_energy = 0.0
    effective_k = None

    for channel_data in channels:
        channel_result = compress_grayscale_svd_precomputed(channel_data, k)
        approx_channels.append(channel_result["approx_float"])

        s = channel_data["s"]
        k_here = channel_result["k"]
        effective_k = k_here if effective_k is None else min(effective_k, k_here)

        total_energy += float(np.sum(s ** 2))
        kept_energy += float(np.sum(s[:k_here] ** 2))

    approx_rgb = np.stack(approx_channels, axis=2)
    original_rgb = np.stack([channel["A"] for channel in channels], axis=2)

    return {
        "k": effective_k,
        "approx_float": approx_rgb,
        "approx_uint8": clip_to_uint8(approx_rgb),
        "frobenius_error": frobenius_error(original_rgb, approx_rgb),
        "relative_frobenius_error": relative_frobenius_error(original_rgb, approx_rgb),
        "storage_ratio": storage_ratio(m, n, effective_k),
        "retained_energy": 0.0 if total_energy == 0 else kept_energy / total_energy,
    }


def validate_metrics(metrics):
    errors = [row["frobenius_error"] for row in metrics]
    rel_errors = [row["relative_frobenius_error"] for row in metrics]
    ratios = [row["storage_ratio"] for row in metrics]
    energies = [row["retained_energy"] for row in metrics]

    return {
        "error_nonincreasing": all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1)),
        "relative_error_nonincreasing": all(
            rel_errors[i] >= rel_errors[i + 1] for i in range(len(rel_errors) - 1)
        ),
        "ratio_nondecreasing": all(ratios[i] <= ratios[i + 1] for i in range(len(ratios) - 1)),
        "energy_nondecreasing": all(
            energies[i] <= energies[i + 1] for i in range(len(energies) - 1)
        ),
    }


def build_eckart_young_table(prepared, k_values):
    s = prepared["s"]
    A = prepared["A"]
    U = prepared["U"]
    Vt = prepared["Vt"]

    rows = []

    for k in k_values:
        A_k = reconstruct_from_svd(U, s, Vt, k)
        computed = frobenius_error(A, A_k)
        theoretical = eckart_young_error(s, k)
        rel_diff = abs(computed - theoretical) / (theoretical + 1e-15)

        rows.append(
            {
                "k": int(k),
                "computed_error": float(computed),
                "theoretical_error": float(theoretical),
                "relative_difference": float(rel_diff),
            }
        )

    return rows
