from pinndicmulti.DIC_importlib import np, tqdm

def DIC3D_Strain_from_Displacement(
    X, Y, Z, U, V, W, mask,
    SmoothLen=5, min_valid=6
):
    """
    3D DIC strain computation using local least squares (NumPy + tqdm)

    Parameters:
        X,Y,Z : (H,W) 坐标
        U,V,W : (H,W) 位移
        mask  : (H,W) 有效区域 (0/1)
        SmoothLen : 窗口尺寸（奇数）
        min_valid : 最少有效点数

    Returns:
        exx, eyy, ezz, exy, exz, eyz
    """

    m = SmoothLen
    if m % 2 == 0:
        m += 1
    hfm = (m - 1) // 2

    H, W_ = X.shape

    # 初始化
    exx = np.full_like(X, np.nan, dtype=np.float64)
    eyy = np.full_like(X, np.nan, dtype=np.float64)
    ezz = np.full_like(X, np.nan, dtype=np.float64)
    exy = np.full_like(X, np.nan, dtype=np.float64)
    exz = np.full_like(X, np.nan, dtype=np.float64)
    eyz = np.full_like(X, np.nan, dtype=np.float64)

    # padding
    pad = ((hfm, hfm), (hfm, hfm))
    Xp = np.pad(X, pad, constant_values=np.nan)
    Yp = np.pad(Y, pad, constant_values=np.nan)
    Zp = np.pad(Z, pad, constant_values=np.nan)

    Up = np.pad(U, pad, constant_values=np.nan)
    Vp = np.pad(V, pad, constant_values=np.nan)
    Wp = np.pad(W, pad, constant_values=np.nan)

    maskp = np.pad(mask, pad, constant_values=0)

    # 主循环（加进度条）
    for j in tqdm.tqdm(range(H), desc="Computing strain"):
        for i in range(W_):

            if mask[j, i] == 0:
                continue

            # 局部窗口
            Xn = Xp[j:j+m, i:i+m].ravel()
            Yn = Yp[j:j+m, i:i+m].ravel()
            Zn = Zp[j:j+m, i:i+m].ravel()

            Un = Up[j:j+m, i:i+m].ravel()
            Vn = Vp[j:j+m, i:i+m].ravel()
            Wn = Wp[j:j+m, i:i+m].ravel()

            Fn = maskp[j:j+m, i:i+m].ravel()

            valid = (Fn == 1) & (~np.isnan(Xn))

            if np.sum(valid) < min_valid:
                continue

            # 构造 A
            A = np.stack([
                np.ones_like(Xn),
                Xn, Yn, Zn
            ], axis=1)[valid]

            Uv = Un[valid]
            Vv = Vn[valid]
            Wv = Wn[valid]

            # 🔥 用正规方程（比lstsq更快）
            AtA = A.T @ A

            try:
                inv = np.linalg.inv(AtA)
            except np.linalg.LinAlgError:
                continue

            a = inv @ (A.T @ Uv)
            b = inv @ (A.T @ Vv)
            c = inv @ (A.T @ Wv)

            # 梯度
            dU_dX, dU_dY, dU_dZ = a[1:]
            dV_dX, dV_dY, dV_dZ = b[1:]
            dW_dX, dW_dY, dW_dZ = c[1:]

            # 应变
            exx[j, i] = dU_dX
            eyy[j, i] = dV_dY
            ezz[j, i] = dW_dZ

            exy[j, i] = 0.5 * (dU_dY + dV_dX)
            exz[j, i] = 0.5 * (dU_dZ + dW_dX)
            eyz[j, i] = 0.5 * (dV_dZ + dW_dY)

    return exx, eyy, ezz, exy, exz, eyz