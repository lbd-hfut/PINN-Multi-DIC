"""
Multi-view point cloud fusion for PINN-Multi-DIC.

Merges pairwise triangulation results into a unified 3D reconstruction.
"""

from pinndicmulti.DIC_importlib import np


def fuse_pairwise_reconstructions(pairwise_pts3D, method="average"):
    """Merge multiple 3D point clouds from pairwise triangulations.

    Parameters
    ----------
    pairwise_pts3D : list of ndarray
        Each element is (N, 3) point cloud from one camera pair.
    method : str
        "average" — arithmetic mean of all clouds
        "median"  — element-wise median (robust to outliers)
        "robust"  — median → reject >2σ outliers → average inliers

    Returns
    -------
    pts3D_merged : ndarray (N, 3)
        Fused point cloud.
    quality : ndarray (N,)
        Per-point quality score (1.0 = perfect agreement, <1.0 = disparity).
        For "robust": fraction of inlier pairs per point.
    """
    if len(pairwise_pts3D) == 0:
        raise ValueError("No point clouds to fuse")
    if len(pairwise_pts3D) == 1:
        return pairwise_pts3D[0], np.ones(pairwise_pts3D[0].shape[0])

    stacked = np.stack(pairwise_pts3D, axis=0)  # (M, N, 3)

    if method == "average":
        merged = np.mean(stacked, axis=0)
        variance = np.var(stacked, axis=0).sum(axis=1)  # (N,)
        max_var = np.max(variance) + 1e-10
        quality = 1.0 - 0.5 * (variance / max_var)
    elif method == "median":
        merged = np.median(stacked, axis=0)
        variance = np.var(stacked, axis=0).sum(axis=1)
        max_var = np.max(variance) + 1e-10
        quality = 1.0 - 0.5 * (variance / max_var)
    elif method == "robust":
        median_cloud = np.median(stacked, axis=0)  # (N, 3)
        # Per-pair deviation from median (Euclidean distance)
        diff = stacked - median_cloud[np.newaxis, :, :]  # (M, N, 3)
        dist = np.linalg.norm(diff, axis=2)  # (M, N)
        # Global outlier threshold: 2× std of all pairwise deviations
        global_std = np.std(dist)
        threshold = max(2.0 * global_std, 1e-6)  # at least 1 µm
        inlier_mask = dist <= threshold  # (M, N)
        inlier_count = np.sum(inlier_mask, axis=0)  # (N,)
        # Average inliers per point; fall back to median if no inliers
        inlier_mask_3d = inlier_mask[:, :, np.newaxis]  # (M, N, 1)
        inlier_sum = np.sum(stacked * inlier_mask_3d, axis=0)  # (N, 3)
        inlier_count_3d = np.maximum(inlier_count[:, np.newaxis], 1)  # (N, 1)
        merged = inlier_sum / inlier_count_3d
        quality = inlier_count.astype(np.float64) / len(pairwise_pts3D)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    return merged, quality


def fuse_with_weights(pairwise_pts3D, weights):
    """Weighted fusion of multiple reconstructions.

    Parameters
    ----------
    pairwise_pts3D : list of ndarray (N, 3)
    weights : list of ndarray (N,)
        Per-point confidence weights for each reconstruction.

    Returns
    -------
    pts3D_merged : ndarray (N, 3)
    """
    if len(pairwise_pts3D) != len(weights):
        raise ValueError("Number of point clouds and weights must match")

    stacked = np.stack(pairwise_pts3D, axis=0)    # (M, N, 3)
    w = np.stack(weights, axis=0)                  # (M, N)
    w_sum = np.sum(w, axis=0, keepdims=True) + 1e-10  # (1, N)

    merged = np.sum(stacked * w[:, :, np.newaxis], axis=0) / w_sum[:, :, np.newaxis]
    return merged
