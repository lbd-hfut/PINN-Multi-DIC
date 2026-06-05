from pinndicmulti.DIC_importlib import os, re, glob, np, cv2, sio

# ==============================
# 1. 读取 MATLAB stereoParams
# ==============================
def load_stereo_params(mat_path):
    data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    # print("MAT keys:", data.keys())
    # 去掉系统字段
    keys = [k for k in data.keys() if not k.startswith("__")]

    # ========= 情况1：扁平结构（你自己保存的） =========
    if all(k in data for k in ["K1", "K2", "dist1", "dist2", "R", "T"]):
        K1 = np.array(data["K1"])
        K2 = np.array(data["K2"])
        dist1 = np.array(data["dist1"]).reshape(-1)
        dist2 = np.array(data["dist2"]).reshape(-1)
        R = np.array(data["R"])
        T = np.array(data["T"]).reshape(3, 1)

        # print("✔ Loaded simple calibration format")
        return K1, K2, dist1, dist2, R, T

    # ========= 情况2：MATLAB stereoParameters =========
    if len(keys) == 1:
        session = data[keys[0]]

        if hasattr(session, "CameraParameters"):
            stereo = session.CameraParameters
        else:
            stereo = session  # 有些文件直接就是 stereoParameters

        # 内参（注意转置）
        K1 = stereo.CameraParameters1.IntrinsicMatrix.T
        K2 = stereo.CameraParameters2.IntrinsicMatrix.T

        # 畸变
        def get_dist(cam):
            radial = np.array(cam.RadialDistortion)
            tangential = getattr(cam, "TangentialDistortion", [0, 0])
            return np.hstack([radial, tangential])

        dist1 = get_dist(stereo.CameraParameters1)
        dist2 = get_dist(stereo.CameraParameters2)

        # 外参
        R = np.array(stereo.RotationOfCamera2)
        T = np.array(stereo.TranslationOfCamera2).reshape(3, 1)

        print("✔ Loaded MATLAB stereoParameters format")
        return K1, K2, dist1, dist2, R, T

    # ========= 兜底 =========
    raise ValueError(f"无法识别的MAT文件结构: {keys}")

# ==============================
# 2. 构造投影矩阵
# ==============================
def build_projection(K1, K2, R, T):
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))
    return P1, P2

# ==============================
# 4. 去畸变
# ==============================
def undistort_points(pts, K, dist):
    pts = pts.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, dist, P=K)
    return undist.reshape(-1, 2)

# ==============================
# 5. 三角重建
# ==============================
def triangulate(P1, P2, ptsL, ptsR):
    ptsL = ptsL.T.astype(np.float64)
    ptsR = ptsR.T.astype(np.float64)

    pts4D = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
    pts3D = pts4D[:3] / pts4D[3]

    return pts3D.T

# ==============================
# 6. mode = 1 left camera first image is reference, others are matched with it
# ==============================
def reconstruct_mode(pts2DL, pts2DR, P1, P2, K1, K2, dist1, dist2):

    x_L = pts2DL[:, 0]
    y_L = pts2DL[:, 1]
    x_R = pts2DR[:, 0]
    y_R = pts2DR[:, 1]

    ptsL = np.stack([x_L, y_L], axis=1)
    ptsR = np.stack([x_R, y_R], axis=1)

    # 去畸变（关键）
    ptsL = undistort_points(ptsL, K1, dist1)
    ptsR = undistort_points(ptsR, K2, dist2)

    pts3D = triangulate(P1, P2, ptsL, ptsR)

    return pts3D

# ==============================
# 7. 主程序
# ==============================
def triangulation(config, Xrerf, Utemporal, Udisparity):

    calibration_path = config.calibration_path
    K1, K2, dist1, dist2, R, T = load_stereo_params(calibration_path)

    P1, P2 = build_projection(K1, K2, R, T)
    pts2DL = Xrerf + Utemporal
    pts2DR = Xrerf + Udisparity
    
    pts3D = reconstruct_mode(pts2DL, pts2DR, P1, P2, K1, K2, dist1, dist2)
    return pts3D


# ==============================
# Multi-camera triangulation
# ==============================

def triangulate_pair(K_i, K_j, dist_i, dist_j, R_i, t_i, R_j, t_j, pts2D_i, pts2D_j):
    """Triangulate 3D points from a camera pair using COLMAP parameters.

    Parameters
    ----------
    K_i, K_j : ndarray (3,3)
        Intrinsic matrices for camera i and j.
    dist_i, dist_j : ndarray (5,)
        Distortion coefficients.
    R_i, t_i : ndarray (3,3), (3,1)
        Camera i rotation and translation (cam_from_world).
    R_j, t_j : ndarray (3,3), (3,1)
        Camera j rotation and translation (cam_from_world).
    pts2D_i, pts2D_j : ndarray (N,2)
        Matched 2D points in camera i and j.

    Returns
    -------
    pts3D : ndarray (N,3)
        Triangulated 3D points in world coordinates.
    """
    pts2D_i = pts2D_i.astype(np.float64)
    pts2D_j = pts2D_j.astype(np.float64)

    # Ensure numeric types (calib may use object arrays for storage)
    K_i = np.asarray(K_i, dtype=np.float64)
    K_j = np.asarray(K_j, dtype=np.float64)
    dist_i = np.asarray(dist_i, dtype=np.float64)
    dist_j = np.asarray(dist_j, dtype=np.float64)
    R_i = np.asarray(R_i, dtype=np.float64)
    t_i = np.asarray(t_i, dtype=np.float64)
    R_j = np.asarray(R_j, dtype=np.float64)
    t_j = np.asarray(t_j, dtype=np.float64)

    # Build projection matrices P = K [R | t]
    Rt_i = np.hstack((R_i, t_i))
    Rt_j = np.hstack((R_j, t_j))
    P_i = K_i @ Rt_i
    P_j = K_j @ Rt_j

    # Undistort points
    pts_i = pts2D_i.reshape(-1, 1, 2).astype(np.float64)
    pts_j = pts2D_j.reshape(-1, 1, 2).astype(np.float64)
    pts_i_undist = cv2.undistortPoints(pts_i, K_i, dist_i, P=K_i).reshape(-1, 2)
    pts_j_undist = cv2.undistortPoints(pts_j, K_j, dist_j, P=K_j).reshape(-1, 2)

    # Triangulate
    pts4D = cv2.triangulatePoints(P_i, P_j, pts_i_undist.T, pts_j_undist.T)
    pts3D = (pts4D[:3] / pts4D[3]).T

    return pts3D


def triangulate_all_pairs(calib, matched_pts_per_camera, roi_coords,
                          return_raw=False):
    """Triangulate 3D points from all camera pairs and return merged cloud.

    Uses camera 0 as reference. For each pair (0, j), triangulates
    the matched points, then transforms all results to the world frame.

    Parameters
    ----------
    calib : dict
        Calibration dict with keys: K_list, dist_list, cam_from_world_R,
        cam_from_world_t, num_cameras.
    matched_pts_per_camera : dict
        dict[cam_idx] = ndarray (N, 2) — matched 2D points in each camera.
        Points are in pixel coordinates relative to camera 0's reference image.
    roi_coords : ndarray (N, 2)
        Reference pixel coordinates in camera 0 (x, y).
    return_raw : bool
        If True, also return the list of raw pairwise point clouds.

    Returns
    -------
    pts3D_merged : ndarray (N, 3)
        Fused 3D point cloud in world coordinates.
    pts3D_all : list of ndarray, only if return_raw=True
        Raw pairwise triangulation results (num_cameras-1 groups of (N,3)).
    """
    num_cameras = calib["num_cameras"]
    K_list = calib["K_list"]
    dist_list = calib["dist_list"]
    R_list = calib["cam_from_world_R"]
    t_list = calib["cam_from_world_t"]

    pts3D_all = []

    # Camera 0 is the primary reference
    # For each camera pair (0, j), triangulate
    pts2D_0 = matched_pts_per_camera[0]  # points in camera 0

    for j in range(1, num_cameras):
        pts2D_j = matched_pts_per_camera[j]

        pts3D = triangulate_pair(
            K_list[0], K_list[j],
            dist_list[0], dist_list[j],
            R_list[0], t_list[0],
            R_list[j], t_list[j],
            pts2D_0, pts2D_j
        )
        pts3D_all.append(pts3D)

    # Simple fusion: average all pairwise reconstructions
    if len(pts3D_all) == 1:
        pts3D_merged = pts3D_all[0]
    else:
        pts3D_stacked = np.stack(pts3D_all, axis=0)  # (num_pairs, N, 3)
        pts3D_merged = np.mean(pts3D_stacked, axis=0)

    if return_raw:
        return pts3D_merged, pts3D_all
    return pts3D_merged


if __name__ == "__main__":
    mat_path = 'C:/01project/PINN-3D-DIC/case/3D/plate_center_load/calibrationSession.mat'
    load_stereo_params(mat_path)
    