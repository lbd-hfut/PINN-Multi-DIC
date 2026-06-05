from pinndicmulti.DIC_importlib import partial, jnp, jax, math, np, os, label, Image
from pinndicmulti.segpinndic.utils.logger import logger

# ============================================
# 线程缓冲区 (用于存储中间计算结果)
# ============================================
class BufferManager:
    SEED_QK = None
    DIC_QK = None
    QKBQKT_def_seed = None
    QKBQKT_def_DIC = None
    fx = None
    fy = None
    refImg = None
    refImg_pad = None
    defImg = None
    defImg_pad = None
    mask = None
    mask_pad = None
    scale_uv = None

def plus_power(x, p):
    return jnp.where(x > 0, x ** p, 0.0)

# 1. 泛化的 B样条基函数及其导数计算
@partial(jax.jit, static_argnums=(1, 2))
def beta_nth(x, n, degree):
    d = degree
    # 静态生成系数和偏移量 (在 JIT 编译期完成)
    coeffs_list = [((-1)**k) * math.comb(d + 1, k) for k in range(d + 2)]
    shifts_list = [(d + 1) / 2.0 - k for k in range(d + 2)]
    
    coeffs = jnp.array(coeffs_list)
    shifts = jnp.array(shifts_list)
    factor = math.factorial(d) // math.factorial(d - n)

    def body(c, s):
        return c * factor * plus_power(x + s, d - n)

    # 向量化计算求和
    val = jnp.sum(jax.vmap(body)(coeffs, shifts), axis=0)
    return val / math.factorial(d)

# 2. 泛化的多项式系数矩阵 QK
@partial(jax.jit, static_argnums=(0,))
def get_QK(degree):
    N = degree + 1
    offset = degree // 2
    # 动态生成局部坐标系 
    # d=5 -> [-2, -1, 0, 1, 2, 3]
    # d=3 -> [-1, 0, 1, 2]
    # d=1 -> [0, 1]
    x = jnp.arange(-offset, -offset + N)
    
    def row(n):
        return ((-1)**n) * beta_nth(x, n, degree) / math.factorial(n)
    return jnp.stack([row(n) for n in range(N)])

# 3. 泛化的 B样条系数滤波器
@partial(jax.jit, static_argnums=(1, 2))
def form_bcoef(img, degree, border=3):
    img = jnp.pad(img, border, mode="edge")
    h, w = img.shape
    
    # 针对 d=1 (双一次插值)，无需预滤波，FFT会自动除以 1
    radius = degree // 2
    x_sample = jnp.arange(-radius, radius + 1)
    kernel_b = beta_nth(x_sample, 0, degree)

    def make_kernel(n):
        k = jnp.zeros(n)
        center_idx = radius
        k = k.at[0].set(kernel_b[center_idx])
        if radius > 0:
            k = k.at[1:radius+1].set(kernel_b[center_idx+1:])
            k = k.at[-radius:].set(kernel_b[:center_idx])
        return jnp.fft.fft(k)

    kx = make_kernel(w)
    ky = make_kernel(h)

    img = jnp.real(jnp.fft.ifft(jnp.fft.fft(img, axis=1) / kx, axis=1))
    img = jnp.real(jnp.fft.ifft(jnp.fft.fft(img, axis=0) / ky[:, None], axis=0))

    return img

# 4. 泛化的子区提取与乘法
@partial(jax.jit, static_argnums=(2, 3))
def get_QK_B_QKT(plot_bcoef, img, degree, border=3, QK=None):
    QKT = QK.T
    offset = degree // 2
    N = degree + 1
    H, W = img.shape
    
    ys, xs = jnp.meshgrid(
        jnp.arange(H),
        jnp.arange(W),
        indexing="ij"
    )

    top  = ys + border - offset
    left = xs + border - offset

    # 动态尺寸切片
    dy = jnp.arange(N)[:, None]    # (N,1)
    dx = jnp.arange(N)[None, :]    # (1,N)

    blocks = plot_bcoef[
        top[..., None, None] + dy,
        left[..., None, None] + dx
    ]                              # (H, W, N, N)

    return jnp.einsum("ij,hwjk,kl->hwil", QK, blocks, QKT)


# 5. 泛化的梯度计算
@partial(jax.jit, static_argnums=(2, 3))
def image_gradient_from_bcoef(ref_bcoef, roi_mask, degree, border=3, QK=None):
    QKT = QK.T
    H, W = roi_mask.shape
    offset = degree // 2
    N = degree + 1

    ys, xs = jnp.meshgrid(
        jnp.arange(H),
        jnp.arange(W),
        indexing="ij"
    )

    top  = ys + border - offset
    left = xs + border - offset

    dy = jnp.arange(N)[:, None]
    dx = jnp.arange(N)[None, :]

    blocks = ref_bcoef[
        top[..., None, None] + dy,
        left[..., None, None] + dx
    ]

    M = jnp.einsum("ij,hwjk,kl->hwil", QK, blocks, QKT)

    fx = M[..., 0, 1]
    fy = M[..., 1, 0]

    return fx, fy

# ============================================
# Buffer 构建函数 (增加 degree 参数)
# ============================================
def build_seed_buffer_jax(img, mask, degree=5):
    BufferManager.SEED_QK = get_QK(degree)
    plot_bcoef = form_bcoef(img, degree)
    logger.info("precomputing seed buffers: gradients")
    BufferManager.fx, BufferManager.fy = image_gradient_from_bcoef(
        plot_bcoef, mask, degree, QK=BufferManager.SEED_QK)
    
def build_DIC_buffer_jax(img, degree=5):
    logger.info("precomputing seed buffers: QKBQKT_def")
    plot_bcoef = form_bcoef(img, degree=5)
    BufferManager.QKBQKT_def_seed = get_QK_B_QKT(plot_bcoef, img, degree=5, QK=BufferManager.SEED_QK)
    logger.info("precomputing DIC buffers: QKBQKT_def")
    if degree != 5:
        BufferManager.DIC_QK = get_QK(degree)
        plot_bcoef = form_bcoef(img, degree)
        BufferManager.QKBQKT_def_DIC = get_QK_B_QKT(plot_bcoef, img, degree, QK=BufferManager.DIC_QK)
    else:
        BufferManager.DIC_QK = BufferManager.SEED_QK
        BufferManager.QKBQKT_def_DIC = BufferManager.QKBQKT_def_seed
    
class ImgDataset2D:
    def __init__(self, DIC_config, Seed_config):
        logger.info(f"Scanning directory for images: {DIC_config.input_dir}")
        image_files = np.array([
            x.path for x in os.scandir(DIC_config.input_dir)
            if x.name.lower().endswith((".bmp", ".png", ".jpg", ".tiff", ".tif"))
        ])
        if image_files.size == 0:
            raise FileNotFoundError(
                f"[ERROR] No image files found in directory: {DIC_config.input_dir} "
                "(supported: .bmp, .png, .jpg, .tiff, .tif)"
            )
        image_files.sort()
        
        # 参考图 & mask
        logger.info(f"Found {len(image_files)} image files. Assuming first is reference and last is mask.")
        self.rfimage_file = image_files[0]
        self.mask_file = image_files[-1]
        # 将参考图像和mask图像存入 BufferManager
        self.coarse_subset_radius = Seed_config.coarse_subset_radius
        BufferManager.refImg = self.open_image(self.rfimage_file)
        BufferManager.refImg_pad = jnp.pad(
            BufferManager.refImg,
            pad_width=self.coarse_subset_radius,
            mode='constant',
            constant_values=False
        )
        mask_bin = self.open_image(self.mask_file) > 0
        labeled, num_labels = label(mask_bin)
        if num_labels == 0:
            raise RuntimeError("Mask 中没有前景像素！")
        ROI_list, ROI_list_pad = [], []
        for comp_id in range(1, num_labels + 1):
            roi_i = (labeled == comp_id)
            roi_i = jnp.array(roi_i, dtype=jnp.bool_)
            roi_i_pad = jnp.pad(
                roi_i,
                pad_width=self.coarse_subset_radius,
                mode='constant',
                constant_values=False
            )
            # 创建单连通域 ROI
            ROI_list.append(roi_i)
            ROI_list_pad.append(roi_i_pad)
        BufferManager.mask = ROI_list
        BufferManager.mask_pad = ROI_list_pad
        logger.info("precomputing seed buffers")
        build_seed_buffer_jax(BufferManager.refImg, mask_bin, degree=5)

        # 变形图像序列
        self.dfimage_files = image_files[1:-1]
        self.spline_degree = getattr(DIC_config, 'spline_degree', 5)

    def __len__(self):
        return len(self.dfimage_files)

    def get_image(self, idx):
        """只负责取图，不产生副作用"""
        BufferManager.defImg = self.open_image(self.dfimage_files[idx])
        BufferManager.defImg_pad = jnp.pad(
            BufferManager.defImg,
            pad_width=self.coarse_subset_radius,
            mode='constant',
            constant_values=False
        )
        build_DIC_buffer_jax(BufferManager.defImg, degree=self.spline_degree)

    @staticmethod
    def open_image(name):
        img = Image.open(name).convert("L")
        return jnp.array(img, dtype=jnp.float32)
    
class ImgDataset3D:
    def __init__(self, DIC_config, Seed_config):
        # 读取相机1，并进行基本的检查
        logger.info(f"Scanning camera1 directory for images: {DIC_config.cam1_dir}")
        cam1_files = np.array([
            x.path for x in os.scandir(DIC_config.cam1_dir)
            if x.name.lower().endswith((".bmp", ".png", ".jpg", ".tiff", ".tif"))
        ])
        if cam1_files.size == 0:
            raise FileNotFoundError(
                f"[ERROR] No image files found in directory: {DIC_config.cam1_dir} "
                "(supported: .bmp, .png, .jpg, .tiff, .tif)"
            )
        cam1_files.sort()
        logger.info(f"Found {len(cam1_files)} image files.")
        # 读取相机2，并进行基本的检查
        logger.info(f"Scanning camera2 directory for images: {DIC_config.cam2_dir}")
        cam2_files = np.array([
            x.path for x in os.scandir(DIC_config.cam2_dir)
            if x.name.lower().endswith((".bmp", ".png", ".jpg", ".tiff", ".tif"))
        ])
        if cam2_files.size == 0:
            raise FileNotFoundError(
                f"[ERROR] No image files found in directory: {DIC_config.cam2_dir} "
                "(supported: .bmp, .png, .jpg, .tiff, .tif)"
            )
        cam2_files.sort()
        logger.info(f"Found {len(cam2_files)} image files.")
        # 检查两组图像数量是否匹配
        if len(cam2_files) != len(cam1_files):
            raise ValueError(
                f"Camera 1 and Camera 2 must have the same number of images. "
                f"Found {len(cam1_files)} in cam1 and {len(cam2_files)} in cam2."
            )
        # 选出参考图像和mask图像
        self.rfimage_file = cam1_files[0]
        self.mask_file = DIC_config.roi_path
        # 将参考图像和mask图像存入 BufferManager
        self.coarse_subset_radius = Seed_config.coarse_subset_radius
        BufferManager.refImg = self.open_image(self.rfimage_file)
        BufferManager.refImg_pad = jnp.pad(
            BufferManager.refImg,
            pad_width=self.coarse_subset_radius,
            mode='constant',
            constant_values=False
        )
        mask_bin = self.open_image(self.mask_file) > 0
        labeled, num_labels = label(mask_bin)
        if num_labels == 0:
            raise RuntimeError("Mask 中没有前景像素！")
        ROI_list, ROI_list_pad = [], []
        for comp_id in range(1, num_labels + 1):
            roi_i = (labeled == comp_id)
            roi_i = jnp.array(roi_i, dtype=jnp.bool_)
            roi_i_pad = jnp.pad(
                roi_i,
                pad_width=self.coarse_subset_radius,
                mode='constant',
                constant_values=False
            )
            # 创建单连通域 ROI
            ROI_list.append(roi_i)
            ROI_list_pad.append(roi_i_pad)
        BufferManager.mask = ROI_list
        BufferManager.mask_pad = ROI_list_pad
        logger.info("precomputing seed buffers")
        build_seed_buffer_jax(BufferManager.refImg, mask_bin, degree=5)
        # 变形图像序列
        self.dfimage_files = []
        self.dfimage_files.append(cam2_files[0])
        for i in range(len(cam2_files)-1):
            self.dfimage_files.append(cam1_files[i+1])
            self.dfimage_files.append(cam2_files[i+1])
        logger.info(f"Deformed sequence built. Total images: {len(self.dfimage_files)}")
        self.spline_degree = getattr(DIC_config, 'spline_degree', 5)
        
    def __len__(self):
        return len(self.dfimage_files)
    
    def get_image(self, idx):
        """只负责取图，不产生副作用"""
        BufferManager.defImg = self.open_image(self.dfimage_files[idx])
        BufferManager.defImg_pad = jnp.pad(
            BufferManager.defImg,
            pad_width=self.coarse_subset_radius,
            mode='constant',
            constant_values=False
        )
        build_DIC_buffer_jax(BufferManager.defImg, degree=self.spline_degree)
    
    @staticmethod
    def open_image(name):
        img = Image.open(name).convert("L")
        return jnp.array(img, dtype=jnp.float32)
        
    
    
def _find_mask(work_dir, mask_path=None):
    """Find mask file: explicit path > work_dir/*.{png,bmp,jpg} > work_dir/image/*."""
    if mask_path and os.path.exists(mask_path):
        return mask_path
    for ext in (".png", ".bmp", ".jpg"):
        for loc in (work_dir, os.path.join(work_dir, "image")):
            p = os.path.join(loc, f"mask{ext}")
            if os.path.exists(p):
                return p
    raise FileNotFoundError(
        f"Mask file not found. Place mask.png/mask.bmp in work_dir/ "
        f"or work_dir/image/, or specify mask_path in config."
    )


class MultiCamDataset:
    """Multi-camera image dataset for PINN-Multi-DIC.

    Auto-discovers N camera folders under work_dir/images/.
    Each camera folder contains: first image = reference, rest = deformed frames.
    The first image of each camera also serves as COLMAP calibration input.
    """

    def __init__(self, work_dir, Seed_config, mask_path=None, mask_camera="1"):
        from pinndicmulti.DIC_config import discover_cameras

        cam_dirs = discover_cameras(work_dir)
        self.num_cameras = len(cam_dirs)

        # Reorder: mask_camera → index 0 (primary reference)
        mask_cam = str(mask_camera)
        if mask_cam not in cam_dirs:
            raise ValueError(
                f"mask_camera='{mask_cam}' not found among camera folders: {cam_dirs}. "
                f"Set mask_camera to the folder name whose reference image the mask aligns with."
            )
        cam_dirs.remove(mask_cam)
        cam_dirs = [mask_cam] + cam_dirs  # mask_camera first
        self.cam_names = cam_dirs
        self.ref_cam_name = mask_cam

        images_dir = os.path.join(work_dir, "images")
        self.coarse_subset_radius = Seed_config.coarse_subset_radius

        # Load all camera image files
        self.cam_ref_files = []      # ref image path per camera
        self.cam_def_files = []      # list of deformed image paths per camera

        for cam_name in cam_dirs:
            cam_path = os.path.join(images_dir, cam_name)
            files = sorted([
                os.path.join(cam_path, f) for f in os.listdir(cam_path)
                if f.lower().endswith((".bmp", ".png", ".jpg", ".tiff", ".tif"))
            ])
            if len(files) < 2:
                raise ValueError(
                    f"Camera {cam_name} needs at least 2 images "
                    f"(1 ref + 1 deformed), found {len(files)}"
                )
            self.cam_ref_files.append(files[0])    # first = ref
            self.cam_def_files.append(files[1:])    # rest = deformed

        # Verify all cameras have same number of deformed frames
        num_def_frames = [len(defs) for defs in self.cam_def_files]
        if len(set(num_def_frames)) > 1:
            raise ValueError(
                f"All cameras must have the same number of deformed frames. "
                f"Got: {dict(zip(cam_dirs, num_def_frames))}"
            )
        self.num_frames = num_def_frames[0]

        # Load mask: explicit path > work_dir/*.{png,bmp,jpg} > work_dir/image/*
        mask_file = _find_mask(work_dir, mask_path)
        logger.info(f"Using mask: {mask_file}")
        mask_bin = MultiCamDataset.open_image(mask_file) > 0
        labeled, num_labels = label(mask_bin)
        if num_labels == 0:
            raise RuntimeError("Mask has no foreground pixels!")

        ROI_list, ROI_list_pad = [], []
        for comp_id in range(1, num_labels + 1):
            roi_i = (labeled == comp_id)
            roi_i = jnp.array(roi_i, dtype=jnp.bool_)
            roi_i_pad = jnp.pad(
                roi_i, pad_width=self.coarse_subset_radius,
                mode='constant', constant_values=False
            )
            ROI_list.append(roi_i)
            ROI_list_pad.append(roi_i_pad)

        BufferManager.mask = ROI_list
        BufferManager.mask_pad = ROI_list_pad

        # Load primary reference camera (mask_camera) as the initial reference
        BufferManager.refImg = self.open_image(self.cam_ref_files[0])
        BufferManager.refImg_pad = jnp.pad(
            BufferManager.refImg, pad_width=self.coarse_subset_radius,
            mode='constant', constant_values=False
        )

        logger.info("precomputing seed buffers")
        build_seed_buffer_jax(BufferManager.refImg, np.asarray(mask_bin), degree=5)

        self.spline_degree = 5
        logger.info(
            f"MultiCamDataset: {self.num_cameras} cameras, "
            f"{self.num_frames} frames, {num_labels} ROIs"
        )

    def __len__(self):
        return self.num_cameras * self.num_frames

    def get_image(self, cam_idx, frame_idx):
        """Load deformed image for camera cam_idx at time frame_idx."""
        def_file = self.cam_def_files[cam_idx][frame_idx]
        BufferManager.defImg = self.open_image(def_file)
        BufferManager.defImg_pad = jnp.pad(
            BufferManager.defImg, pad_width=self.coarse_subset_radius,
            mode='constant', constant_values=False
        )
        build_DIC_buffer_jax(BufferManager.defImg, degree=self.spline_degree)

    def set_ref_image(self, cam_idx):
        """Switch BufferManager to use camera cam_idx as reference."""
        BufferManager.refImg = self.open_image(self.cam_ref_files[cam_idx])
        BufferManager.refImg_pad = jnp.pad(
            BufferManager.refImg, pad_width=self.coarse_subset_radius,
            mode='constant', constant_values=False
        )

    @staticmethod
    def open_image(name):
        img = Image.open(name).convert("L")
        return jnp.array(img, dtype=jnp.float32)

    def set_mask(self, mask_list):
        """Replace BufferManager.mask (e.g. with a warped mask for this camera)."""
        pad = self.coarse_subset_radius
        BufferManager.mask = [jnp.asarray(m, dtype=jnp.bool_) for m in mask_list]
        BufferManager.mask_pad = [
            jnp.pad(m, pad_width=pad, mode='constant', constant_values=False)
            for m in BufferManager.mask
        ]


def warp_mask_via_disparity(mask_0, u_disp, v_disp, shape):
    """Forward-warp a reference-camera mask to another view using disparity.

    Args:
        mask_0: (H, W) binary array — mask in reference camera
        u_disp: (H, W) u displacement from ref camera to target camera
        v_disp: (H, W) v displacement from ref camera to target camera
        shape: (H, W) target image shape

    Returns:
        warped_mask: (H, W) binary array
    """
    from scipy.ndimage import binary_dilation

    H, W = shape
    ys, xs = np.where(mask_0)
    xj = xs + u_disp[ys, xs]
    yj = ys + v_disp[ys, xs]

    xj_int = np.round(xj).astype(int)
    yj_int = np.round(yj).astype(int)

    valid = (xj_int >= 0) & (xj_int < W) & (yj_int >= 0) & (yj_int < H)
    xj_int = xj_int[valid]
    yj_int = yj_int[valid]

    if len(xj_int) == 0:
        return np.zeros(shape, dtype=bool)

    mask_warped = np.zeros(shape, dtype=bool)
    mask_warped[yj_int, xj_int] = True
    mask_warped = binary_dilation(mask_warped, iterations=3)

    return mask_warped


if __name__ == "__main__":

    from segpinndic.DIC_config import seed_config_txt, DIC_2D_config_txt
    seed_config_path = "./config/Seed_Configuration.txt"
    dic_config_path = "./config/PINN-DIC-2D.txt"

    DIC_config = DIC_2D_config_txt(dic_config_path, verbose=False)
    Seed_config = seed_config_txt(seed_config_path, verbose=False)
    
    ImgData = ImgDataset2D(DIC_config, Seed_config)
    ImgData.get_image(0)
