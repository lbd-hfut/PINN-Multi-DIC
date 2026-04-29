from pinndicmulti.DIC_importlib import *
from pinndicmulti.segpinndic.DIC_readImg import BufferManager

# ============================================
# 数据结构定义
# ============================================
class RectangleROI:
    """圆形感兴趣区域"""
    def __init__(self):
        self.x = 0              # 中心x坐标
        self.y = 0              # 中心y坐标
        self.radius = 0         # 半径
        self.mask = None        # 掩膜 (2D array)
        self.region = None      # 区域信息
        self.X_flat = None
        self.Y_flat = None
        
class GNState(NamedTuple):
    defvec: jax.Array      # (6,)
    diffnorm: jax.Array   # scalar
    corrcoef: jax.Array   # scalar
    it: jax.Array         # int scalar
    flag: jax.Array       # int scalar

FAILED = False
SUCCESS = True
# ============================================
# 自动生成种子点位置
# ============================================
class seed_generator:
    def __init__(self, config: SimpleNamespace):
        
        self.config = config
        self.ROI_LIST = BufferManager.mask
        self.seed_points_list = self.sample_kmeans() # np.darray: shape[N,2]
        
    def sample_kmeans(self):
        n_points = self.config.seeds_number
        seed_points_list = []
        for mask in self.ROI_LIST:
            ys, xs = np.nonzero(mask)
            pts = np.column_stack([xs, ys])          # (N,2) 所有 ROI 前景像素坐标
            if len(pts) < n_points:
                raise ValueError("ROI 中像素数量不足")
            # --- 特殊情况：n=1 ---
            if n_points == 1:
                idx = np.random.randint(0, len(pts))
                x, y = pts[idx]
                return [(int(x), int(y))]
            # --- 正常情况：K-means 聚类 ---
            kmeans = KMeans(n_clusters=n_points, n_init='auto').fit(pts)
            centers = np.rint(kmeans.cluster_centers_).astype(int)
            # --- 处理每个中心点 ---
            H, W = mask.shape
            seed_points = []
            for x, y in centers:
                # 中心点合法（落在 ROI 内）
                if 0 <= x < W and 0 <= y < H and mask[y, x]:
                    seed_points.append((int(x), int(y)))
                    continue
                # 中心点无效 → 随机从 ROI 内重新采样
                idx = np.random.randint(0, len(pts))
                xr, yr = pts[idx]
                seed_points.append((int(xr), int(yr)))
            seed_points_list.append(jnp.array(seed_points, dtype=jnp.int32))
        return seed_points_list

# ============================================
# 主类：计算种子点
# ============================================
class CalcSeeds:
    def __init__(self, Seed_config: SimpleNamespace):
        self.max_workers = Seed_config.max_workers
        self.coarse_subset_radius = Seed_config.coarse_subset_radius
        self.fine_subset_radius = Seed_config.fine_subset_radius
        self.max_iterations = Seed_config.max_iterations
        self.cutoff_diffnorm = Seed_config.cutoff_diffnorm
        self.max_iterations = Seed_config.max_iterations
        self.lambda_reg = Seed_config.lambda_reg
        
        seed_Gen = seed_generator(Seed_config)
        self.seed_points_list = seed_Gen.seed_points_list
        self.method = Seed_config.method
        
    # ============================================
    # Step 1: 计算单个种子点
    # ============================================
    def cal_point(self, cy: int, cx: int, num_thread: int = 0):
        # Step 1: 获取矩形感兴趣区域
        coarse_rectroi = self._get_coarse_retroi(cx, cy)
        
        # Step 2: 整像素匹配 - 获取初值
        defvector_init, max_cc = initialguess(coarse_rectroi, num_thread)
        if defvector_init is None:
            return None, 0, 0
        
        if self.method == "Integer_pixels":
            paramvector = defvector_init + [max_cc]
            return paramvector, 0, 0
        
        # Step 3: 获取矩形感兴趣区域
        fine_rectroi = self._get_fine_retroi(cx, cy)
        
        # Step 4: 亚像素精化
        flage, defvector, corrcoef, diffnorm, num_iterations = iterativesearch_py(
            defvector_init, fine_rectroi, 
            max_iter = self.max_iterations,
            cutoff_diffnorm = self.cutoff_diffnorm,
            lambda_reg = self.lambda_reg
            )
        if flage == FAILED:
            return None, 0, 0
        
        # Step 5: 组织输出 [x, y, u, v, corrcoef]
        paramvector = [defvector[0]] + [defvector[1]] + [corrcoef]
        return paramvector, diffnorm, num_iterations
    
    # ============================================
    # 主入口：处理多个种子点
    # ============================================
    def analyze(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # 处理各种子点
        seed_pos_list, seed_uv_list = [], []
        for seed_points in self.seed_points_list:
            valid_seed_pos = []
            valid_seed_uv  = []
            for i, (seed_x, seed_y) in enumerate(
                tqdm.tqdm(
                    seed_points, 
                    desc="Processing seeds", 
                    total=len(seed_points))
            ):
                paramvector, diffnorm, num_iterations = self.cal_point(seed_y, seed_x, 0)
                if paramvector is not None:
                    u, v = paramvector[0], paramvector[1]
                    valid_seed_pos.append([seed_x, seed_y])
                    valid_seed_uv.append([u, v])
            seed_pos = jnp.asarray(valid_seed_pos, dtype=jnp.float32)
            seed_uv = jnp.asarray(valid_seed_uv,  dtype=jnp.float32) # IQR 去除异常值
            seed_pos, seed_uv = self.remove_outliers_iqr(seed_pos, seed_uv)
            seed_pos_list.append(seed_pos)
            seed_uv_list.append(seed_uv)
        return seed_pos_list, seed_uv_list
    
    def remove_outliers_iqr(self, seed_pos, seed_uv, factor=1.5):
        uv = np.array(seed_uv)

        q1 = np.percentile(uv, 25, axis=0)
        q3 = np.percentile(uv, 75, axis=0)
        iqr = q3 - q1

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        mask = np.all((uv >= lower) & (uv <= upper), axis=1)

        return jnp.asarray(seed_pos[mask]), jnp.asarray(seed_uv[mask])
    
    def _get_coarse_retroi(self, x: int, y: int) -> RectangleROI:
        """获取感兴趣区域"""
        rectroi = RectangleROI()
        
        rectroi.radius = self.coarse_subset_radius
        py = y + self.coarse_subset_radius
        px = x + self.coarse_subset_radius
        y0, y1 = py - self.coarse_subset_radius, py + self.coarse_subset_radius + 1
        x0, x1 = px - self.coarse_subset_radius, px + self.coarse_subset_radius + 1
        
        rectroi.x = px
        rectroi.y = py
        
        region = 0
        for roi in BufferManager.mask_pad:
            if roi[py, px]:
                rectroi.region = region
                break
            else:
                region += 1
            
        # 构建掩膜
        rectroi.mask = BufferManager.mask_pad[region][y0:y1, x0:x1]
        return rectroi
    
    def _get_fine_retroi(self, x: int, y: int) -> RectangleROI:
        """获取感兴趣区域"""
        rectroi = RectangleROI()
        rectroi.x = x
        rectroi.y = y
        rectroi.radius = self.fine_subset_radius
        py = y + self.coarse_subset_radius
        px = x + self.coarse_subset_radius
        y0, y1 = py - self.fine_subset_radius, py + self.fine_subset_radius + 1
        x0, x1 = px - self.fine_subset_radius, px + self.fine_subset_radius + 1
        
        region = 0
        for roi in BufferManager.mask_pad:
            if roi[py, px]:
                rectroi.region = region
                break
            else:
                region += 1
        rectroi.mask = jnp.asarray(
            BufferManager.mask_pad[region][y0:y1, x0:x1], dtype=jnp.float32)
        x_offsets = jnp.arange(
            -self.fine_subset_radius,
            self.fine_subset_radius + 1,
            dtype=jnp.int32
        )
        y_offsets = jnp.arange(
            -self.fine_subset_radius,
            self.fine_subset_radius + 1,
            dtype=jnp.int32
        )
        xv, yv = jnp.meshgrid(x_offsets, y_offsets)  # shape (S,S)
        rectroi.X_flat = xv.reshape(-1)   # (subset_area,)
        rectroi.Y_flat = yv.reshape(-1)
        return rectroi
        
    
# ===================== 辅助函数：图像处理 =====================
# -------------------------
# 整像素匹配 - 多尺度NCC
# -------------------------
def initialguess(rectroi: RectangleROI, num_thread: int) -> List:
    # 定义多尺度参数
    reduction_factors = []
    if rectroi.radius // 15 > 0:
        reduction_factors = [rectroi.radius // 15, 0]
    else:
        reduction_factors = [0]
    
    disp_ncc = None
    disp_prev = None  # 空表示第一次迭代
    
    # 迭代多个尺度
    for reduction_factor in reduction_factors:
        disp_ncc, max_cc = ncc(reduction_factor, disp_prev, rectroi, num_thread)
        if disp_ncc is None:
            return None, -1
        # 传递结果给下一次迭代
        disp_prev = disp_ncc
    # 返回最终整像素位移
    defvector_init = [float(disp_ncc[0]), float(disp_ncc[1])]
    return defvector_init, max_cc

# -------------------------
# 归一化互相关计算
# -------------------------
def ncc(reduction_multigrid: int, disp_prev: List, 
        rectroi: RectangleROI, num_thread: int) -> List:
    # -------- 第一部分：获取降采样当前图像 --------
    if disp_prev is None:
        # 第一次迭代：使用整个降采样图像
        reduction_factor = reduction_multigrid + 1
        # 降采样
        cur_reduced = _downsample_image(
            BufferManager.defImg_pad, reduction_factor
        )
        up_cur_multigrid = 0
        left_cur_multigrid = 0
    else:
        # 后续迭代：在上次结果附近搜索
        x_subset = rectroi.x + disp_prev[0]
        y_subset = rectroi.y + disp_prev[1]
        # 截断因子 - 逐次缩小搜索范围
        truncfactor = reduction_multigrid + 1.5
        # 定义搜索范围
        up_cur = max(int(y_subset - truncfactor * rectroi.radius), 0)
        down_cur = min(int(y_subset + truncfactor * rectroi.radius), 
                        BufferManager.defImg_pad.shape[0] - 1)
        left_cur = max(int(x_subset - truncfactor * rectroi.radius), 0)
        right_cur = min(int(x_subset + truncfactor * rectroi.radius), 
                        BufferManager.defImg_pad.shape[1] - 1)
        # 对齐到网格并降采样该区域
        reduction_factor = reduction_multigrid + 1
        cur_reduced, up_cur_multigrid, left_cur_multigrid = \
            _extract_region_downsampled(
                BufferManager.defImg_pad, left_cur, right_cur, up_cur, down_cur,
                reduction_factor, x_subset, y_subset
            )
            
    # -------- 第二部分：获取降采样参考子集 --------
    ref_reduced, mask_reduced = _extract_reference_subset(reduction_multigrid, rectroi)
    
    # -------- 第三部分：匹配 --------
    ncc_map = ncc_conv2d(cur_reduced, ref_reduced, mask_reduced)
    best_x, best_y, max_ncc = best_ncc_from_map(ncc_map)
    
    # -------- 第四部分：反演位移 --------
    if max_ncc < 0.2:
        return None, -1
    else:
        # 从降采样坐标反演到原始像素坐标
        ref_h, ref_w = ref_reduced.shape
        half_w = ref_w // 2
        half_h = ref_h // 2
        x_match_reduced = best_x + half_w
        y_match_reduced = best_y + half_h
        x_match = left_cur_multigrid + x_match_reduced * reduction_factor
        y_match = up_cur_multigrid + y_match_reduced * reduction_factor
        u = x_match - rectroi.x
        v = y_match - rectroi.y
        return [u, v], max_ncc

@jax.jit
def ncc_conv2d(cur_img, ref_patch, mask, eps=1e-6):
    """
    cur_img   : (Hc, Wc)
    ref_patch : (Hr, Wr)
    mask      : (Hr, Wr), bool
    """

    # ---------- reshape for conv ----------
    cur = cur_img[None, None, :, :]          # (N=1, C=1, H, W)
    mask_f = mask.astype(jnp.float32)
    Hr, Wr = ref_patch.shape
    n = jnp.sum(mask_f)
    # ---------- reference preprocessing ----------
    ref_mean = jnp.sum(ref_patch * mask_f) / n
    ref_zero = (ref_patch - ref_mean) * mask_f
    denom_ref = jnp.sum(ref_zero ** 2)
    # ---------- kernels ----------
    k_mask = mask_f[None, None, :, :]
    k_ref  = ref_zero[None, None, :, :]
    # ---------- convolutions ----------
    S_C = lax.conv_general_dilated(
        cur, k_mask,
        window_strides=(1, 1),
        padding="VALID"
    )
    S_C2 = lax.conv_general_dilated(
        cur * cur, k_mask,
        window_strides=(1, 1),
        padding="VALID"
    )
    S_RC = lax.conv_general_dilated(
        cur, k_ref,
        window_strides=(1, 1),
        padding="VALID"
    )
    # ---------- statistics ----------
    mean_C = S_C / n
    var_C = S_C2 - n * mean_C ** 2
    std_C = jnp.sqrt(jnp.maximum(var_C, 0.0))
    denom = jnp.sqrt(denom_ref * var_C + eps)
    ncc = S_RC / denom
    # 无效位置
    valid = (std_C > eps) & (denom_ref > eps)
    ncc = jnp.where(valid, ncc, -jnp.inf)
    return ncc[0, 0]   # (Hc-Hr+1, Wc-Wr+1)

@jax.jit
def best_ncc_from_map(ncc_map):
    """
    ncc_map : (H, W)
    """
    flat_idx = jnp.argmax(ncc_map)
    max_ncc = ncc_map.reshape(-1)[flat_idx]

    W = ncc_map.shape[1]
    best_y = flat_idx // W
    best_x = flat_idx % W

    return best_x, best_y, max_ncc

def _downsample_image(image: jnp.ndarray, factor: int) -> jnp.ndarray:
    """降采样图像"""
    return image[::factor, ::factor].copy()

def _extract_region_downsampled(image, left, right, up, down, 
                                reduction_factor, x_subset, y_subset):
    """提取并降采样区域"""
    # 对齐到网格
    up_aligned = y_subset - int((y_subset - up) / reduction_factor) * reduction_factor
    left_aligned = x_subset - int((x_subset - left) / reduction_factor) * reduction_factor
    down_aligned = y_subset + int((down - y_subset) / reduction_factor) * reduction_factor
    right_aligned = x_subset + int((right - x_subset) / reduction_factor) * reduction_factor
    # 提取并降采样
    region = image[up_aligned:down_aligned+1, left_aligned:right_aligned+1]
    reduced = region[::reduction_factor, ::reduction_factor].copy()
    return reduced, up_aligned, left_aligned

def _extract_reference_subset(reduction_multigrid: int, 
                                rectroi: RectangleROI) -> Tuple:
    """提取并降采样参考子集"""
    reduction_factor = reduction_multigrid + 1
    ref_subset = _extract_region(
        BufferManager.refImg_pad, rectroi.x, rectroi.y, rectroi.radius
    )
    # 降采样
    ref_reduced = ref_subset[::reduction_factor, ::reduction_factor].copy()
    mask_reduced = rectroi.mask[::reduction_factor, ::reduction_factor]
    return ref_reduced, mask_reduced

def _extract_region(image: jnp.ndarray, cx: int, cy: int, 
                                radius: int) -> jnp.ndarray:
    """提取圆形区域"""
    size = 2 * radius + 1
    y_start = max(0, cy - radius)
    x_start = max(0, cx - radius)
    y_end = min(image.shape[0], cy + radius + 1)
    x_end = min(image.shape[1], cx + radius + 1)
    
    region = image[y_start:y_end, x_start:x_end]
    return region

# -------------------------
# Newton (IC-GN) 函数入口
# -------------------------
'''
iterativesearch_py        ← Python，取数据
    ↓
iterativesearch_jax       ← @jit，主循环（lax.while_loop）
    ├─ precompute_gn_terms
    ├─ newton_step
    │    ├─ interpqbs
    │    ├─ safe_cholesky
    │    └─ inverse_compositional_update
'''
def iterativesearch_py(
    defvector_init: jnp.ndarray,
    rectroi: RectangleROI,
    max_iter: int,
    cutoff_diffnorm: float,
    lambda_reg: float
):
    # ---------- ROI & 几何 ----------
    xc = rectroi.x
    yc = rectroi.y

    mask = rectroi.mask.reshape(-1).astype(jnp.float32)
    valid_idx = jnp.where(mask>0)[0]  
    dx = rectroi.X_flat[valid_idx].astype(jnp.int32)
    dy = rectroi.Y_flat[valid_idx].astype(jnp.int32)

    # ---------- Reference image & gradient ----------
    X = xc + dx
    Y = yc + dy
    
    # ---------- numpy → jax ----------
    defv0 = jnp.asarray(
        jnp.concatenate([
            jnp.asarray(defvector_init, jnp.float32),
            jnp.zeros(4, dtype=jnp.float32)
        ])
    )

    f = BufferManager.refImg[Y, X]
    fx = BufferManager.fx[Y, X]
    fy = BufferManager.fy[Y, X]
    
    n = jnp.sum(mask)
    fm = jnp.sum(f) / n
    diff = (f - fm)
    deltaf = jnp.sqrt(jnp.sum(diff * diff))

    if deltaf < lambda_reg:
        return FAILED, defv0, -1, 0, 0
    else: 
        deltaf_inv = 1.0 / deltaf
        df_dp = jnp.stack([
            fx, fy, dx*fx, dx*fy, dy*fx, dy*fy
        ], axis=1)
        H = 2.0 * (deltaf_inv**2) * (df_dp.T @ df_dp)
        
        try:
            cholesky_G = jnp.linalg.cholesky(H)
            positivedef = True
        except RuntimeError:
            positivedef = False  # 非正定
            
        if positivedef:
            defv = defv0
            for iter in range(max_iter):
                defv, diffnorm, corr, ok = newton_step(
                    defv, BufferManager.QKBQKT_def_seed,
                    xc, yc, dx, dy, mask,
                    f, fm, deltaf_inv,
                    df_dp, cholesky_G, lambda_reg
                )
                if diffnorm < cutoff_diffnorm:
                    return SUCCESS, defv, corr, diffnorm, iter
                if ok == FAILED:
                    return FAILED, defv0, corr, diffnorm, iter
            return SUCCESS, defv, corr, diffnorm, iter
        else:
            return FAILED, defvector_init, -1.0, -1.0, 0

@jax.jit
def newton_step(
    defv, QKBQKT_def_seed,
    xc, yc, dx, dy, mask,
    f, fm, deltaf_inv,
    df_dp, cholesky_G, lambda_reg
):
    u = defv[0] + defv[2] * dx + defv[4] * dy
    v = defv[1] + defv[3] * dx + defv[5] * dy

    Xw = xc + dx + u
    Yw = yc + dy + v

    g, oob = interpqbs(Xw, Yw, QKBQKT_def_seed)

    gm = jnp.sum(g) / (jnp.sum(mask))
    diffg = (g - gm)
    deltag = jnp.sqrt(jnp.sum(diffg * diffg))

    valid = deltag > lambda_reg
    deltag_inv = jnp.where(valid, 1.0 / deltag, 0.0)

    r = ((f - fm) * deltaf_inv - (g - gm) * deltag_inv)
    corrcoef = jnp.sum(r * r)

    grad = 2 * deltaf_inv * (r @ df_dp)
    
    y = jnp.linalg.solve(cholesky_G, grad)
    x = jnp.linalg.solve(cholesky_G.T, y)
    delta = -x
    
    # ---------- update ----------
    diffnorm = jnp.linalg.norm(delta)
    defv_new = inverse_compositional_update(defv, delta)
    
    # ---------- correlation (optional) ---------
    ok = jnp.isfinite(diffnorm) & valid
    return defv_new, diffnorm, corrcoef, ok


# -------------------------
# B-spline 插值
# -------------------------
@jax.jit
def interpqbs(xs, ys, QKBQKT_def_seed):
    H, W = QKBQKT_def_seed.shape[:2]

    xs_floor = jnp.floor(xs).astype(jnp.int32)
    ys_floor = jnp.floor(ys).astype(jnp.int32)

    xs_oob = (xs_floor < 0) | (xs_floor >= W)
    ys_oob = (ys_floor < 0) | (ys_floor >= H)
    mask = xs_oob | ys_oob

    xs_floor = jnp.clip(xs_floor, 0, W - 1)
    ys_floor = jnp.clip(ys_floor, 0, H - 1)

    # (N,6,6)
    QK_B_QKT = QKBQKT_def_seed[ys_floor, xs_floor]

    xd = xs - xs_floor
    yd = ys - ys_floor

    powers = jnp.arange(6)
    x_vec = xd[:, None] ** powers[None, :]
    y_vec = yd[:, None] ** powers[None, :]

    tmp = jnp.einsum("ni,nij->nj", y_vec, QK_B_QKT)
    values = jnp.einsum("ni,ni->n", tmp, x_vec)

    return values, mask

# -------------------------
# 逆组合更新
# -------------------------
@jax.jit
def inverse_compositional_update(defv, delta):
    U, V, dudx, dvdx, dudy, dvdy = defv
    du, dv, d_dudx, d_dvdx, d_dudy, d_dvdy = delta

    M_old = jnp.array([
        [1 + dudx, dudy, U],
        [dvdx, 1 + dvdy, V],
        [0, 0, 1]
    ])

    M_d = jnp.array([
        [1 + d_dudx, d_dudy, du],
        [d_dvdx, 1 + d_dvdy, dv],
        [0, 0, 1]
    ])

    M_new = M_old @ jnp.linalg.inv(M_d)

    return jnp.array([
        M_new[0, 2],
        M_new[1, 2],
        M_new[0, 0] - 1,
        M_new[1, 0],
        M_new[0, 1],
        M_new[1, 1] - 1
    ])
    
def Seed_match_visualization(refImg, defImg, xy, uv, output_dir, basename, idx):
    """
    Visualize seed point matching between refImg and defImg.
    xy, uv can be:
        • ndarray [N, 2]
        • list of ndarrays [[N1,2], [N2,2], ...]
    """

    # --- Convert xy/uv to stacked array if they are lists ---
    if isinstance(xy, list):
        xy = [np.asarray(a) for a in xy]
        xy = np.vstack(xy) if len(xy) > 0 else np.zeros((0, 2))
    else:
        xy = np.asarray(xy)
    if isinstance(uv, list):
        uv = [np.asarray(a) for a in uv]
        uv = np.vstack(uv) if len(uv) > 0 else np.zeros((0, 2))
    else:
        uv = np.asarray(uv)

    # Ensure shapes match
    assert xy.shape == uv.shape, "xy and uv must have the same shape."

    # --- Ensure grayscale images are displayed properly ---
    refImg = np.asarray(refImg)
    defImg = np.asarray(defImg)
    if len(refImg.shape) == 2:
        ref_disp = cv2.cvtColor(refImg, cv2.COLOR_GRAY2BGR)
        def_disp = cv2.cvtColor(defImg, cv2.COLOR_GRAY2BGR)
    else:
        ref_disp = refImg.copy()
        def_disp = defImg.copy()

    # Compute matched points in deformed image
    xy_def = xy + uv

    # --- Create visualization canvas ---
    H, W = ref_disp.shape[:2]
    canvas = np.zeros((H, W * 2, 3), dtype=np.uint8)
    canvas[:, :W] = cv2.normalize(ref_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canvas[:, W:] = cv2.normalize(def_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- Draw matching points ---
    for (x1, y1), (x2, y2) in zip(xy, xy_def):
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2) + W, int(y2))

        cv2.circle(canvas, pt1, 4, (0, 255, 0), -1)
        cv2.circle(canvas, pt2, 4, (0, 0, 255), -1)
        cv2.line(canvas, pt1, pt2, (255, 255, 0), 1)

    # Display using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(canvas[..., ::-1])
    ax.set_title(f"Seed Point Matching (Frame {idx})")
    ax.axis("off")

    # Determine save directory
    result_dir = os.path.join(output_dir, "seed")
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, basename + ".png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Visualization saved to: {save_path}")

if __name__ == "__main__":
    from segpinndic.DIC_config import seed_config_txt, DIC_2D_config_txt
    from segpinndic.DIC_readImg import ImgDataset
    from scipy.io import savemat
    seed_config_path = "./config/Seed_Configuration.txt"
    dic_config_path = "./config/PINN-DIC-2D.txt"

    DIC_config = DIC_2D_config_txt(dic_config_path, verbose=False)
    Seed_config = seed_config_txt(seed_config_path, verbose=False)
    
    ImgData = ImgDataset(DIC_config, Seed_config)
    ImgData.get_image(0)
    
    SeedCalculator = CalcSeeds(Seed_config)
    seed_pos, seed_uv = SeedCalculator.analyze()    
    
    Seed_match_visualization(
        BufferManager.refImg*255, 
        BufferManager.defImg*255,
        seed_pos, seed_uv, DIC_config.output_dir, 'seed',0
    )
    
    BufferManager.scale_uv = [jnp.asarray((
            (jnp.max(a[:,0]) + jnp.min(a[:,0]))/2,
            (jnp.max(a[:,1]) + jnp.min(a[:,1]))/2,
            (jnp.max(a[:,0]) - jnp.min(a[:,0]))/2,
            (jnp.max(a[:,1]) - jnp.min(a[:,1]))/2)) for a in seed_uv]
    
    for i, scale_uv in enumerate(BufferManager.scale_uv):
        print(f"roi_{i} scale uv: {scale_uv}")
        print(f"umax: {jnp.max(seed_uv[i][:,0])}, v_max: {jnp.max(seed_uv[i][:,1])}")
        print(f"umin: {jnp.min(seed_uv[i][:,0])}, v_min: {jnp.min(seed_uv[i][:,1])}")
    # savemat("buffer1.mat", 
    #         {
    #             "QKBQKT_def_seed": BufferManager.QKBQKT_def_seed,
    #             "fx": BufferManager.fx,
    #             "fy": BufferManager.fy,
    #             "QK": BufferManager.SEED_QK,
    #         }
    #     )
    
    