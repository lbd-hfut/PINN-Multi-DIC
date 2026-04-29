from pinndicmulti.DIC_importlib import cv2, np, glob, os, Path, savemat

def stereo_calibrate(
    calibrate_config,
    DIC_config,
):
    """
    双目标定（支持棋盘格 / 圆点阵）
    """
    
    # ===== 读取参数 =====
    left_imgs_dir = calibrate_config.calibrate1_dir
    right_imgs_dir = calibrate_config.calibrate2_dir
    pattern_type = calibrate_config.pattern_type
    pattern_size = calibrate_config.pattern_size
    square_size = calibrate_config.length
    visualize = calibrate_config.visualize
    result_dir = os.path.join(DIC_config.output_dir, "calibration")
    os.makedirs(result_dir, exist_ok=True)
    calibration_path = DIC_config.calibration_path

    # ===== 世界坐标 =====
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    left_imgs = load_images_from_dir(left_imgs_dir)
    right_imgs = load_images_from_dir(right_imgs_dir)

    assert len(left_imgs) == len(right_imgs), "左右图数量不一致"

    # ===== 逐张处理 =====
    for fl, fr in zip(left_imgs, right_imgs):
        img_l = cv2.imread(fl)
        img_r = cv2.imread(fr)

        # 左图
        if img_l is None:
            raise ValueError(f"读取失败: {fl}")
        if len(img_l.shape) == 2:
            gray_l = img_l  # 已经是灰度图
        else:
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

        # 右图
        if img_r is None:
            raise ValueError(f"读取失败: {fr}")
        if len(img_r.shape) == 2:
            gray_r = img_r
        else:
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if pattern_type == "chessboard":
            ret_l, corners_l = cv2.findChessboardCornersSB(gray_l, pattern_size)
            ret_r, corners_r = cv2.findChessboardCornersSB(gray_r, pattern_size)

        elif pattern_type == "circles":
            ret_l, corners_l = cv2.findCirclesGrid(gray_l, pattern_size)
            ret_r, corners_r = cv2.findCirclesGrid(gray_r, pattern_size)

        else:
            raise ValueError("pattern_type must be chessboard or circles")

        if ret_l and ret_r:
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

            if visualize:
                save_name = os.path.splitext(os.path.basename(fl))[0] + ".png"  # 用左图名字
                save_path = os.path.join(result_dir, save_name)
                draw_stereo_matches(img_l, img_r, corners_l, corners_r, save_path)

    cv2.destroyAllWindows()

    # ===== 单目标定 =====
    ret_l, K1, dist1, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, gray_l.shape[::-1], None, None
    )

    ret_r, K2, dist2, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, gray_r.shape[::-1], None, None
    )

    # ===== 双目标定 =====
    flags = cv2.CALIB_FIX_INTRINSIC

    ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        K1, dist1,
        K2, dist2,
        gray_l.shape[::-1],
        flags=flags
    )

    # ===== 投影矩阵（关键）=====
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K2 @ np.hstack((R, T))

    result = {
        "K1": K1,
        "dist1": dist1,
        "K2": K2,
        "dist2": dist2,
        "R": R,
        "T": T,
        "P1": P1,
        "P2": P2,
        "error": ret
    }
    
    savemat(calibration_path, result)
    
def load_images_from_dir(img_dir):
    extensions = ["*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    img_dir = Path(img_dir)
    files = []
    for ext in extensions:
        files.extend(img_dir.glob(ext))
    return sorted([str(f) for f in files])

def draw_stereo_matches(img_l, img_r, corners_l, corners_r, save_path=None):
    """
    将左右图拼接，并绘制特征点连线
    """
    # 转为 (N,2)
    pts_l = corners_l.reshape(-1, 2)
    pts_r = corners_r.reshape(-1, 2)

    # 拼接图像（横向）
    h1, w1 = img_l.shape[:2]
    h2, w2 = img_r.shape[:2]

    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)

    canvas[:h1, :w1] = img_l
    canvas[:h2, w1:w1+w2] = img_r

    # 画点和连线
    for (x1, y1), (x2, y2) in zip(pts_l, pts_r):
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2) + w1, int(y2))  # 右图要加偏移

        color = tuple(np.random.randint(0, 255, 3).tolist())

        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1)

    # 显示
    # cv2.imshow("stereo matches", canvas)
    # cv2.waitKey(200)

    # 保存
    if save_path is not None:
        cv2.imwrite(save_path, canvas)

    return canvas

if __name__ == "__main__":
    from pinndicmulti.DIC_config import DIC_3D_config_txt, calibrate_config_txt
    DIC_config = DIC_3D_config_txt("./config/PINN-DIC-3D.txt", verbose=False)
    calibrate_config = calibrate_config_txt(
        "./config/Calibration_Configuration.txt", verbose=False)
    
    stereo_calibrate(
        calibrate_config,
        DIC_config,
    )