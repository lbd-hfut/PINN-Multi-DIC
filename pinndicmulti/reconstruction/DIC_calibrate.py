"""
Multi-camera self-calibration using COLMAP Structure-from-Motion.

Uses the first image of each camera folder as calibration input.
Outputs camera intrinsics, distortion, and poses in .mat format.
"""

from pinndicmulti.DIC_importlib import cv2, np, os, savemat, Path
from pinndicmulti.DIC_config import discover_cameras, get_work_subdir


def colmap_calibrate_multi_camera(work_dir):
    """Self-calibrate N cameras using COLMAP SfM on reference images.

    Each camera's first image (work_dir/images/camN/001.jpg) is used
    as calibration input. COLMAP runs SfM to recover intrinsics and
    relative poses for all cameras.

    Results saved to work_dir/calibration/cameras.mat:
        num_cameras: int
        K_list: list of 3x3 matrices
        dist_list: list of distortion vectors
        cam_from_world_R: list of 3x3 rotation matrices
        cam_from_world_t: list of 3x1 translation vectors
        P_list: list of 3x4 projection matrices
        camera_models: list of model name strings
        camera_params_list: list of raw param vectors

    Parameters
    ----------
    work_dir : str
        Root working directory containing images/ subdirectory.
    """
    import pycolmap

    cam_dirs = discover_cameras(work_dir)
    num_cameras = len(cam_dirs)
    images_dir = get_work_subdir(work_dir, "images")
    calib_dir = get_work_subdir(work_dir, "calibration")
    os.makedirs(calib_dir, exist_ok=True)

    # ----- 1. Collect reference images from each camera -----
    ref_images = {}  # cam_name -> ref image path
    for cam_name in cam_dirs:
        cam_path = os.path.join(images_dir, cam_name)
        files = sorted([
            f for f in os.listdir(cam_path)
            if f.lower().endswith((".bmp", ".png", ".jpg", ".tiff", ".tif"))
        ])
        if not files:
            raise FileNotFoundError(f"No images in {cam_path}")
        ref_images[cam_name] = os.path.join(cam_path, files[0])

    # ----- 2. Copy ref images to temp dir with camera prefix -----
    colmap_image_dir = os.path.join(calib_dir, "colmap_calib_images")
    os.makedirs(colmap_image_dir, exist_ok=True)

    for cam_name, ref_path in ref_images.items():
        basename = os.path.basename(ref_path)
        new_name = f"{cam_name}_{basename}"
        new_path = os.path.join(colmap_image_dir, new_name)
        img = cv2.imread(ref_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {ref_path}")
        cv2.imwrite(new_path, img)

    # ----- 3. Run COLMAP SfM -----
    database_path = os.path.join(calib_dir, "colmap.db")
    sfm_path = os.path.join(calib_dir, "colmap_sfm")

    # Clean previous runs
    import shutil
    if os.path.exists(database_path):
        os.remove(database_path)
    if os.path.exists(sfm_path):
        shutil.rmtree(sfm_path)
    os.makedirs(sfm_path, exist_ok=True)

    pycolmap.set_random_seed(0)

    # Feature extraction
    pycolmap.extract_features(
        database_path, colmap_image_dir,
        extraction_options={"sift": {"max_num_features": 8192, "first_octave": 0}}
    )

    # Exhaustive matching
    pycolmap.match_exhaustive(
        database_path,
        matching_options={"sift": {"cross_check": True}}
    )

    # Incremental SfM
    reconstructions = pycolmap.incremental_mapping(
        database_path, colmap_image_dir, sfm_path,
        options={
            "ba_global_max_refinements": 5,
            "min_num_matches": 15,
            "multiple_models": False,
            "min_model_size": max(3, num_cameras),
            "min_focal_length_ratio": 0.1,
            "max_focal_length_ratio": 10.0,
        }
    )

    if not reconstructions:
        raise RuntimeError(
            "COLMAP SfM failed. Ensure calibration images have sufficient "
            "texture and overlap between camera views."
        )

    rec = reconstructions[0]

    # ----- 4. Group images by camera and extract params -----
    cam_image_ids = {cam_name: [] for cam_name in cam_dirs}
    for image_id, image in rec.images.items():
        for cam_name in cam_dirs:
            if image.name.startswith(f"{cam_name}_"):
                cam_image_ids[cam_name].append(image_id)
                break

    K_list = []
    dist_list = []
    cam_from_world_R = []
    cam_from_world_t = []
    P_list = []
    camera_models = []
    camera_params_list = []

    for cam_name in cam_dirs:
        ids = cam_image_ids[cam_name]
        if not ids:
            raise RuntimeError(
                f"Camera {cam_name}: no registered images in COLMAP output. "
                f"Check that the reference image has sufficient texture."
            )

        # Get camera intrinsics from the first registered image
        cam_id = rec.images[ids[0]].camera_id
        camera = rec.cameras[cam_id]

        K, dist = _extract_camera_params(camera)
        K_list.append(K)
        dist_list.append(dist)
        camera_models.append(str(camera.model))
        camera_params_list.append(np.array(camera.params, dtype=np.float64))

        # Use the first registered image's pose as this camera's world pose
        # pycolmap 4.x: cam_from_world is a method, call it to get Rigid3d
        cfw = rec.images[ids[0]].cam_from_world
        if callable(cfw):
            cfw = cfw()
        cam_from_world_R.append(cfw.rotation.matrix().astype(np.float64))
        cam_from_world_t.append(cfw.translation.astype(np.float64).reshape(3, 1))

        # Build projection matrix: P = K [R | t]
        Rt = np.hstack((cam_from_world_R[-1], cam_from_world_t[-1]))
        P_list.append(K @ Rt)

    # ----- 5. Save results -----
    result = {
        "num_cameras": num_cameras,
        "K_list": np.array(K_list, dtype=object),
        "dist_list": np.array(dist_list, dtype=object),
        "cam_from_world_R": np.array(cam_from_world_R, dtype=object),
        "cam_from_world_t": np.array(cam_from_world_t, dtype=object),
        "P_list": np.array(P_list, dtype=object),
        "camera_models": np.array(camera_models, dtype=object),
        "camera_params_list": np.array(camera_params_list, dtype=object),
        "cam_names": np.array(cam_dirs, dtype=object),
        "num_registered_images": rec.num_reg_images(),
    }

    mat_path = os.path.join(calib_dir, "cameras.mat")
    savemat(mat_path, result)

    print(f"[COLMAP] Multi-camera calibration saved to: {mat_path}")
    print(f"  Cameras: {num_cameras} ({', '.join(cam_dirs)})")
    for i, cam_name in enumerate(cam_dirs):
        print(f"  {cam_name}: model={camera_models[i]}, "
              f"f={K_list[i][0,0]:.1f}, "
              f"num_images={len(cam_image_ids[cam_name])}")
    print(f"  Registered images: {rec.num_reg_images()}/{num_cameras}")

    return result


def _extract_camera_params(camera):
    """Extract K matrix and distortion from COLMAP camera object.

    Returns
    -------
    K : np.ndarray (3, 3)
        Intrinsic matrix
    dist : np.ndarray (5,)
        Distortion coefficients [k1, k2, p1, p2, k3]
    """
    params = camera.params
    model = str(camera.model)

    if model in ("SIMPLE_PINHOLE",):
        f, cx, cy = params[0], params[1], params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)

    elif model in ("PINHOLE",):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)

    elif model in ("SIMPLE_RADIAL",):
        f, cx, cy, k1 = params[0], params[1], params[2], params[3]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    elif model in ("RADIAL",):
        f, cx, cy, k1, k2 = params[0], params[1], params[2], params[3], params[4]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

    else:
        # Fallback: use calibration_matrix() with zero distortion
        K = camera.calibration_matrix().astype(np.float64)
        dist = np.zeros(5, dtype=np.float64)

    return K, dist


# ============================================================
# Legacy compatibility wrappers
# ============================================================

def stereo_calibrate(calibrate_config, DIC_config):
    """Legacy wrapper: calls colmap_calibrate_multi_camera with work_dir.

    Kept for backward compatibility with old config format that had
    calibrate1_dir, calibrate2_dir, output_dir, calibration_path keys.
    """
    # Extract work_dir from DIC_config
    work_dir = getattr(DIC_config, 'work_dir', None)
    if work_dir is None:
        # Fallback: try to derive from calibrate_config
        raise ValueError(
            "Legacy config format no longer supported. "
            "Please use the new work_dir-based config."
        )
    return colmap_calibrate_multi_camera(work_dir)


def load_images_from_dir(img_dir):
    """Load sorted image file paths from directory."""
    extensions = ["*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    img_dir = Path(img_dir)
    files = []
    for ext in extensions:
        files.extend(img_dir.glob(ext))
    return sorted([str(f) for f in files])
