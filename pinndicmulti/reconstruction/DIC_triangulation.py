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

if __name__ == "__main__":
    mat_path = 'C:/01project/PINN-3D-DIC/case/3D/plate_center_load/calibrationSession.mat'
    load_stereo_params(mat_path)
    