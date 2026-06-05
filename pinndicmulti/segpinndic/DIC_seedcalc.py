from pinndicmulti.DIC_importlib import *
from pinndicmulti.segpinndic.DIC_readImg import BufferManager
from pinndicmulti.segpinndic.utils.logger import logger

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
            # --- 处理每个中心点，含纹理检查 ---
            H, W = mask.shape
            seed_points = []
            min_std = getattr(self.config, 'min_texture_std', 5)
            coarse_r = self.config.coarse_subset_radius

            for x, y in centers:
                # 边界检查
                if 0 <= x < W and 0 <= y < H and mask[y, x]:
                    # 纹理检查：在参考图像 pad 中取子区，计算局部标准差
                    px = x + coarse_r
                    py = y + coarse_r
                    y0 = py - coarse_r
                    y1 = py + coarse_r + 1
                    x0 = px - coarse_r
                    x1 = px + coarse_r + 1
                    local_patch = BufferManager.refImg_pad[y0:y1, x0:x1]
                    local_std = float(jnp.std(local_patch))
                    if local_std >= min_std:
                        seed_points.append((int(x), int(y)))
                        continue
                    # 纹理不足 → 在当前簇内随机搜索更好的位置
                    cluster_dist = np.sqrt((pts[:, 0] - x)**2 + (pts[:, 1] - y)**2)
                    cluster_mask = cluster_dist < np.percentile(cluster_dist, 30)
                    candidates = pts[cluster_mask]
                    if len(candidates) > 0:
                        best_std = 0.0
                        best_pt = None
                        for cx, cy in candidates[:50]:  # 最多尝试 50 个候选
                            cpx, cpy = int(cx) + coarse_r, int(cy) + coarse_r
                            cy0, cy1 = cpy - coarse_r, cpy + coarse_r + 1
                            cx0, cx1 = cpx - coarse_r, cpx + coarse_r + 1
                            if (cy0 >= 0 and cx0 >= 0 and
                                cy1 <= BufferManager.refImg_pad.shape[0] and
                                cx1 <= BufferManager.refImg_pad.shape[1]):
                                cp = BufferManager.refImg_pad[cy0:cy1, cx0:cx1]
                                c_std = float(jnp.std(cp))
                                if c_std > best_std:
                                    best_std = c_std
                                    best_pt = (int(cx), int(cy))
                        if best_pt is not None and best_std >= min_std:
                            seed_points.append(best_pt)
                            continue
                    # 纹理检查失败，但仍保留（避免种子数不足）
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
        self.corrcoef_threshold = getattr(Seed_config, 'corrcoef_threshold', 2.0)
        self.min_texture_std  = getattr(Seed_config, 'min_texture_std', 5)
        self.ncc_threshold    = getattr(Seed_config, 'ncc_threshold', 0.6)

        seed_Gen = seed_generator(Seed_config)
        self.seed_points_list = seed_Gen.seed_points_list
        self.method = Seed_config.method
        self.n_seeds = Seed_config.seeds_number
        
    # ============================================
    # Step 1: 计算单个种子点
    # ============================================
    def cal_point(self, cy: int, cx: int, num_thread: int = 0):
        # Step 1: 获取矩形感兴趣区域
        coarse_rectroi = self._get_coarse_retroi(cx, cy)

        # Step 1.5: 纹理预检查 — 低纹理区域拒绝匹配
        roi_mask = coarse_rectroi.mask
        ref_patch = _extract_region(
            BufferManager.refImg_pad, coarse_rectroi.x, coarse_rectroi.y,
            coarse_rectroi.radius
        )
        if roi_mask.shape != ref_patch.shape:
            ref_patch = ref_patch[:roi_mask.shape[0], :roi_mask.shape[1]]
        n_coarse = jnp.sum(roi_mask)
        if n_coarse > 0:
            ref_patch_m = ref_patch.astype(jnp.float32) * roi_mask.astype(jnp.float32)
            ref_mean = jnp.sum(ref_patch_m) / n_coarse
            ref_var = jnp.sum((ref_patch_m - ref_mean * roi_mask) ** 2) / n_coarse
            if jnp.sqrt(jnp.maximum(ref_var, 0.0)) < self.min_texture_std:
                return None, 0, 0, 'texture'

        # Step 2: 整像素匹配 - 获取初值
        defvector_init, max_cc = initialguess(coarse_rectroi, num_thread, self.ncc_threshold)
        if defvector_init is None:
            return None, 0, 0, 'ncc'

        if self.method == "Integer_pixels":
            paramvector = defvector_init + [max_cc]
            return paramvector, 0, 0, 'ok'

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
            return None, 0, 0, 'icgn'

        # Step 4.5: ZNSSD 相关系数过滤 (NCORR cutoff_corrcoef)
        if corrcoef > self.corrcoef_threshold:
            return None, 0, 0, 'corrcoef'

        # Step 5: 组织输出 [u, v, corrcoef]
        paramvector = [defvector[0]] + [defvector[1]] + [corrcoef]
        return paramvector, diffnorm, num_iterations, 'ok'
    
    # ============================================
    # 主入口：处理多个种子点
    # ============================================
    def analyze(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.method == "SIFT":
            return self._analyze_sift()

        # 处理各种子点
        seed_pos_list, seed_uv_list = [], []
        for seed_points in self.seed_points_list:
            valid_seed_pos = []
            valid_seed_uv  = []
            n_total = len(seed_points)
            fail_counts = {'texture': 0, 'ncc': 0, 'icgn': 0, 'corrcoef': 0}
            fail_positions = {'texture': [], 'ncc': [], 'icgn': [], 'corrcoef': []}
            for i, (seed_x, seed_y) in enumerate(
                tqdm.tqdm(
                    seed_points,
                    desc="Processing seeds",
                    total=n_total)
            ):
                paramvector, diffnorm, num_iterations, reason = self.cal_point(seed_y, seed_x, 0)
                if reason == 'ok':
                    u, v = paramvector[0], paramvector[1]
                    valid_seed_pos.append([seed_x, seed_y])
                    valid_seed_uv.append([u, v])
                else:
                    fail_counts[reason] += 1
                    fail_positions[reason].append((int(seed_x), int(seed_y)))
            n_pass = len(valid_seed_pos)
            logger.info(
                f"Seed matching: {n_pass}/{n_total} passed "
                f"(texture:{fail_counts['texture']} ncc:{fail_counts['ncc']} "
                f"icgn:{fail_counts['icgn']} corrcoef:{fail_counts['corrcoef']})"
            )
            if fail_positions['ncc']:
                ncc_xy = fail_positions['ncc'][:5]
                logger.info(f"  NCC fail examples (x,y): {ncc_xy}")
            if fail_positions['icgn']:
                icgn_xy = fail_positions['icgn'][:5]
                logger.info(f"  IC-GN fail examples (x,y): {icgn_xy}")
            # ---- MAD-based outlier rejection for displacement ----
            if n_pass >= 8:
                uv_arr = jnp.asarray(valid_seed_uv, dtype=jnp.float32)
                median_u = jnp.median(uv_arr[:, 0])
                median_v = jnp.median(uv_arr[:, 1])
                mad_u = jnp.median(jnp.abs(uv_arr[:, 0] - median_u))
                mad_v = jnp.median(jnp.abs(uv_arr[:, 1] - median_v))
                thresh_u = 4.5 * mad_u + 1e-6
                thresh_v = 4.5 * mad_v + 1e-6
                inlier_mask = (
                    (jnp.abs(uv_arr[:, 0] - median_u) < thresh_u) &
                    (jnp.abs(uv_arr[:, 1] - median_v) < thresh_v)
                )
                n_outliers = jnp.sum(~inlier_mask)
                if n_outliers > 0:
                    logger.info(f"  MAD outlier rejection: {n_outliers} outliers removed")
                valid_seed_pos = [valid_seed_pos[i] for i in range(n_pass)
                                  if inlier_mask[i]]
                valid_seed_uv  = [valid_seed_uv[i] for i in range(n_pass)
                                  if inlier_mask[i]]
            elif n_pass > 0:
                logger.warning(
                    f"  Only {n_pass} seeds passed, skipping outlier rejection"
                )
            seed_pos = jnp.asarray(valid_seed_pos, dtype=jnp.float32)
            seed_uv = jnp.asarray(valid_seed_uv,  dtype=jnp.float32)
            seed_pos_list.append(seed_pos)
            seed_uv_list.append(seed_uv)
        return seed_pos_list, seed_uv_list
    
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

    def _analyze_sift(self) -> Tuple[List, List]:
        """SIFT 特征匹配 + 网格优选种子点。

        将 ROI 外接矩形等分为 seeds_number 个网格单元，
        每个单元内选 SIFT 匹配质量（response / distance）最好的点。
        """
        import cv2

        ref_img = np.asarray(BufferManager.refImg, dtype=np.uint8)
        def_img = np.asarray(BufferManager.defImg, dtype=np.uint8)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(ref_img, None)
        kp2, des2 = sift.detectAndCompute(def_img, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            logger.warning("[SIFT] Not enough keypoints")
            empty = [jnp.zeros((0, 2), dtype=jnp.float32) for _ in BufferManager.mask]
            return empty, empty

        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),  # FLANN_INDEX_KDTREE
            dict(checks=50))
        raw_matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        logger.info(f"[SIFT] {len(good)}/{len(raw_matches)} matches passed Lowe's ratio test")

        if len(good) == 0:
            empty = [jnp.zeros((0, 2), dtype=jnp.float32) for _ in BufferManager.mask]
            return empty, empty

        # 提取匹配点坐标与质量
        pts_ref = np.array([kp1[m.queryIdx].pt for m in good])      # (N, 2)
        pts_def = np.array([kp2[m.trainIdx].pt for m in good])
        responses = np.array([kp1[m.queryIdx].response for m in good])
        distances = np.array([m.distance for m in good])
        quality = responses / (distances + 1e-6)

        uv = pts_def - pts_ref

        seed_pos_list, seed_uv_list = [], []
        for roi_id, mask in enumerate(BufferManager.mask):
            mask_np = np.asarray(mask)
            H, W = mask_np.shape
            ys_roi, xs_roi = np.where(mask_np)
            if len(xs_roi) == 0:
                seed_pos_list.append(jnp.zeros((0, 2), dtype=jnp.float32))
                seed_uv_list.append(jnp.zeros((0, 2), dtype=jnp.float32))
                continue

            # 筛选 ROI 内的匹配点
            pts_int = np.rint(pts_ref).astype(int)
            inside = np.array([
                0 <= x < W and 0 <= y < H and mask_np[y, x]
                for x, y in pts_int
            ])
            idx_roi = np.where(inside)[0]
            if len(idx_roi) == 0:
                logger.warning(f"[SIFT] No matches inside ROI {roi_id}")
                seed_pos_list.append(jnp.zeros((0, 2), dtype=jnp.float32))
                seed_uv_list.append(jnp.zeros((0, 2), dtype=jnp.float32))
                continue

            pts_roi = pts_ref[idx_roi]
            uv_roi = uv[idx_roi]
            qual_roi = quality[idx_roi]

            # 网格划分
            xmin, xmax = xs_roi.min(), xs_roi.max()
            ymin, ymax = ys_roi.min(), ys_roi.max()
            aspect = (xmax - xmin + 1) / (ymax - ymin + 1)
            n_cols = max(1, int(round(np.sqrt(self.n_seeds * aspect))))
            n_rows = max(1, int(round(self.n_seeds / n_cols)))
            cols_edges = np.linspace(xmin, xmax + 1, n_cols + 1)
            rows_edges = np.linspace(ymin, ymax + 1, n_rows + 1)

            # 每个网格单元选最佳匹配
            cell_seed_pos = []
            cell_seed_uv = []
            for r in range(n_rows):
                for c in range(n_cols):
                    x0, x1 = int(cols_edges[c]), int(cols_edges[c + 1])
                    y0, y1 = int(rows_edges[r]), int(rows_edges[r + 1])
                    in_cell = (
                        (pts_roi[:, 0] >= x0) & (pts_roi[:, 0] < x1) &
                        (pts_roi[:, 1] >= y0) & (pts_roi[:, 1] < y1)
                    )
                    idx_cell = np.where(in_cell)[0]
                    if len(idx_cell) == 0:
                        continue
                    best = idx_cell[np.argmax(qual_roi[idx_cell])]
                    cell_seed_pos.append(pts_roi[best])
                    cell_seed_uv.append(uv_roi[best])

            if len(cell_seed_pos) < 3:
                logger.warning(f"[SIFT] ROI {roi_id}: only {len(cell_seed_pos)} cells filled, skip")
                seed_pos_list.append(jnp.zeros((0, 2), dtype=jnp.float32))
                seed_uv_list.append(jnp.zeros((0, 2), dtype=jnp.float32))
                continue

            # MAD 离群值剔除
            uv_arr = np.array(cell_seed_uv, dtype=np.float32)
            median_u = np.median(uv_arr[:, 0])
            median_v = np.median(uv_arr[:, 1])
            mad_u = np.median(np.abs(uv_arr[:, 0] - median_u))
            mad_v = np.median(np.abs(uv_arr[:, 1] - median_v))
            thresh = 4.5
            inlier = (
                (np.abs(uv_arr[:, 0] - median_u) < thresh * mad_u + 1e-6) &
                (np.abs(uv_arr[:, 1] - median_v) < thresh * mad_v + 1e-6)
            )
            n_out = np.sum(~inlier)
            if n_out > 0:
                logger.info(f"[SIFT] ROI {roi_id}: MAD removed {n_out} outliers")

            cell_seed_pos = [cell_seed_pos[i] for i in range(len(cell_seed_pos)) if inlier[i]]
            cell_seed_uv = [cell_seed_uv[i] for i in range(len(cell_seed_uv)) if inlier[i]]

            logger.info(f"[SIFT] ROI {roi_id}: {len(cell_seed_pos)} seed points selected "
                        f"({n_rows}x{n_cols} grid)")

            seed_pos_list.append(jnp.asarray(cell_seed_pos, dtype=jnp.float32))
            seed_uv_list.append(jnp.asarray(cell_seed_uv, dtype=jnp.float32))

        return seed_pos_list, seed_uv_list


# ===================== 辅助函数：图像处理 =====================
# -------------------------
# 整像素匹配 - 多尺度NCC
# -------------------------
def initialguess(rectroi: RectangleROI, num_thread: int, ncc_threshold: float = 0.6) -> List:
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
        disp_ncc, max_cc = ncc(reduction_factor, disp_prev, rectroi, num_thread, ncc_threshold)
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
        rectroi: RectangleROI, num_thread: int, ncc_threshold: float = 0.6) -> List:
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
    if max_ncc < ncc_threshold: # 阈值过低，匹配失败
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

    print(f"[OK] Visualization saved to: {save_path}")

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
    
    