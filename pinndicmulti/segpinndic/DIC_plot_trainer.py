from pinndicmulti.DIC_importlib import jax, jnp, np, plt, matplotlib, plt, os, Axes3D

from pinndicmulti.segpinndic.utils.other import colors
from pinndicmulti.segpinndic.utils.logger import logger

def _lim(v, factor=1.1):
    mi, ma = v.min(0), v.max(0)
    c = (mi+ma)/2
    w = factor*(ma-mi)/2
    return (c-w, c+w)

def _plot_setup(x_batch_test, u_exact):
    # get general setup for plotting
    xlim, ulim = _lim(x_batch_test), _lim(u_exact)
    return xlim, ulim

def _to_numpy(f):
    # converts jnp arrays to np arrays
    def wrapper(*args):
        args = jax.tree_map(lambda a: np.array(a) if isinstance(a, jnp.ndarray) else a, args)
        return f(*args)
    return wrapper

@_to_numpy
def plot_1D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)

    f = plt.figure(figsize=(8,4*10/3))

    # plot domain + x_batch
    plt.subplot(4,1,1)
    plt.title(f"[{i}] Domain decomposition")
    plt.scatter(x_batch[:,0], 0.1*np.ones_like(x_batch)[:,0], alpha=0.5, color="k", s=40)
    decomposition.plot(all_params, active=active, create_fig=False)
    plt.xlim(*xlim)

    plt.subplot(4,1,2)
    plt.title(f"[{i}] POU window functions")
    for im in range(all_params["static"]["decomposition"]["m"]):
        plt.plot(x_batch_test[:,0], ws_test[im,:,0], color=colors[im])
    plt.xlim(*xlim)

    # plot full + individual solutions
    plt.subplot(4,1,3)
    plt.title(f"[{i}] Full and individual solutions")
    for im in range(all_params["static"]["decomposition"]["m"]):
        plt.plot(x_batch_test[:,0], us_test[im,:,0], color=colors[im])
    plt.plot(x_batch_test[:,0], u_exact[:,0], lw=4, color="tab:grey", label="Ground truth")
    plt.plot(x_batch_test[:,0], u_test[:,0], color="k", label="FBPINN")
    plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ulim)

    # plot raw solutions
    plt.subplot(4,1,4)
    plt.title(f"[{i}] Raw solutions")
    for im in range(all_params["static"]["decomposition"]["m"]):
        plt.plot(x_batch_test[:,0], us_raw_test[im,:,0], color=colors[im])
    plt.xlim(*xlim)

    plt.tight_layout()

    return (("test",f),)

@_to_numpy
def plot_1D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)

    f = plt.figure(figsize=(8,10))

    # plot x_batch
    plt.subplot(3,1,1)
    plt.title(f"[{i}] Training points")
    plt.scatter(x_batch[:,0], 0.1*np.ones_like(x_batch)[:,0], alpha=0.5, color="k", s=40)
    plt.xlim(*xlim)

    # plot full solution
    plt.subplot(3,1,2)
    plt.title(f"[{i}] Full solution")
    plt.plot(x_batch_test[:,0], u_exact[:,0], lw=4, color="tab:grey", label="Ground truth")
    plt.plot(x_batch_test[:,0], u_test[:,0], color="k", label="PINN")
    plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ulim)

    # plot raw solution
    plt.subplot(3,1,3)
    plt.title(f"[{i}] Raw solution")
    plt.plot(x_batch_test[:,0], u_raw_test[:,0], color="k")
    plt.xlim(*xlim)

    plt.tight_layout()

    return (("test",f),)

def _plot_test_im(u_test, xlim, ulim, n_test, it=None):
    u_test = u_test.reshape(n_test)
    if it is not None:
        u_test = u_test[:,:,it]# for 3D
    plt.imshow(u_test.T,# transpose as jnp.meshgrid uses indexing="ij"
               origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
               cmap="viridis", vmin=ulim[0], vmax=ulim[1])
    plt.colorbar()
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")

@_to_numpy
def plot_2D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch_test.min(0), x_batch_test.max(0)

    f = plt.figure(figsize=(8,10))

    # plot domain + x_batch
    plt.subplot(3,2,1)
    plt.title(f"[{i}] Domain decomposition")
    plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    decomposition.plot(all_params, active=active, create_fig=False)
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")

    # plot full solutions
    plt.subplot(3,2,2)
    plt.title(f"[{i}] Difference")
    _plot_test_im(u_exact - u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,3)
    plt.title(f"[{i}] Full solution")
    _plot_test_im(u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,4)
    plt.title(f"[{i}] Ground truth")
    _plot_test_im(u_exact, xlim0, ulim, n_test)

    # plot raw hist
    plt.subplot(3,2,5)
    plt.title(f"[{i}] Raw solutions")
    plt.hist(us_raw_test.flatten(), bins=100, label=f"{us_raw_test.min():.1f}, {us_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)

@_to_numpy
def plot_2D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch.min(0), x_batch.max(0)

    f = plt.figure(figsize=(8,10))

    # plot x_batch
    plt.subplot(3,2,1)
    plt.title(f"[{i}] Training points")
    plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")

    # plot full solution
    plt.subplot(3,2,2)
    plt.title(f"[{i}] Difference")
    _plot_test_im(u_exact - u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,3)
    plt.title(f"[{i}] Full solution")
    _plot_test_im(u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,4)
    plt.title(f"[{i}] Ground truth")
    _plot_test_im(u_exact, xlim0, ulim, n_test)

    # plot raw hist
    plt.subplot(3,2,5)
    plt.title(f"[{i}] Raw solution")
    plt.hist(u_raw_test.flatten(), bins=100, label=f"{u_raw_test.min():.1f}, {u_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)

_plotters = {
    "FBPINN":{1: plot_1D_FBPINN,
              2: plot_2D_FBPINN
        },
    "PINN":  {1: plot_1D_PINN,
              2: plot_2D_PINN
        },
    }

def plot(trainer, dims, *args):
    "Plots FBPINN and PINN results"

    nx = dims[1]
    if trainer in _plotters and nx in _plotters[trainer]:
        return _plotters[trainer][nx](*args)
    else:
        return ()# TODO: add higher-dim plots
    
def zero_to_nan(matrix):
    matrix = np.array(matrix)
    matrix[matrix == 0] = np.nan
    return matrix

def _nonzero_minmax(arr):
    arr = np.asarray(arr, dtype=float)
    arr_nz = arr[arr != 0]
    arr_nz = arr_nz[~np.isnan(arr_nz)]
    if arr_nz.size == 0:
        return None, None
    return np.min(arr_nz), np.max(arr_nz)


def plot_seed_prediction(u_roi, v_roi, mask, save_dir, label):
    """Plot predicted u, v displacement fields after seed training.

    Args:
        u_roi: (N,) u displacement at ROI points
        v_roi: (N,) v displacement at ROI points
        mask: (H, W) boolean ROI mask
        save_dir: base figure output directory (e.g. c.fig_out_dir)
        label: filename label (e.g. "pinn_roi0_pair1")
    """
    u_roi = np.array(u_roi)
    v_roi = np.array(v_roi)

    u_img = np.full(mask.shape, np.nan, dtype=np.float32)
    v_img = np.full(mask.shape, np.nan, dtype=np.float32)
    ys, xs = np.where(mask)
    u_img[ys, xs] = u_roi
    v_img[ys, xs] = v_roi

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

    im1 = ax1.imshow(u_img, cmap='jet', interpolation='nearest')
    ax1.set_title("u (seed prediction)", fontsize=10)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(v_img, cmap='jet', interpolation='nearest')
    ax2.set_title("v (seed prediction)", fontsize=10)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()

    seed_dir = os.path.join(save_dir, "seed")
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)
    file_path = os.path.join(seed_dir, f"seed_{label}.png")
    fig.savefig(file_path, bbox_inches='tight')
    logger.info(f"Seed prediction saved to {file_path}")
    plt.close(fig)


def result_uv_strain_plot(u, v, exx, exy, eyy,
                          layout=[2,3], WH=[5,4],
                          save_dir=None, filename=None):
    """
    layout:
    [ u   v   empty ]
    [ exx eyy exy   ]
    """

    # --- min/max (ignore zero) ---
    u_min, u_max = _nonzero_minmax(u)
    v_min, v_max = _nonzero_minmax(v)
    exxmin, exxmax = _nonzero_minmax(exx)
    if exxmin == exxmax:
        exxmin, exxmax = exxmin-0.01, exxmax+0.01
    eyymin, eyymax = _nonzero_minmax(eyy)
    if eyymin == eyymax:
        eyymin, eyymax = eyymin-0.01, eyymax+0.01
    exymin, exymax = _nonzero_minmax(exy)
    if exymin == exymax:
        exymin, exymax = exymin-0.01, exymax+0.01

    # --- zero -> nan ---
    u = zero_to_nan(u)
    v = zero_to_nan(v)
    exx = zero_to_nan(exx)
    eyy = zero_to_nan(eyy)
    exy = zero_to_nan(exy)

    # --- figure ---
    plt.figure(figsize=(WH[1]*layout[1], WH[0]*layout[0]), dpi=200)

    normu   = matplotlib.colors.Normalize(vmin=u_min,   vmax=u_max)
    normv   = matplotlib.colors.Normalize(vmin=v_min,   vmax=v_max)
    normexx = matplotlib.colors.Normalize(vmin=exxmin, vmax=exxmax)
    normeyy = matplotlib.colors.Normalize(vmin=eyymin, vmax=eyymax)
    normexy = matplotlib.colors.Normalize(vmin=exymin, vmax=exymax)

    # ===== Row 1: displacement =====
    plt.subplot(layout[0], layout[1], 1)
    plt.imshow(u, cmap='jet', interpolation='nearest', norm=normu)
    plt.colorbar()
    plt.title("u", fontsize=10)
    plt.axis('off')

    plt.subplot(layout[0], layout[1], 2)
    plt.imshow(v, cmap='jet', interpolation='nearest', norm=normv)
    plt.colorbar()
    plt.title("v", fontsize=10)
    plt.axis('off')

    # 第3个位置留空
    plt.subplot(layout[0], layout[1], 3)
    plt.axis('off')

    # ===== Row 2: strain =====
    plt.subplot(layout[0], layout[1], 4)
    plt.imshow(exx, cmap='jet', interpolation='nearest', norm=normexx)
    plt.colorbar()
    plt.title("exx", fontsize=10)
    plt.axis('off')

    plt.subplot(layout[0], layout[1], 5)
    plt.imshow(eyy, cmap='jet', interpolation='nearest', norm=normeyy)
    plt.colorbar()
    plt.title("eyy", fontsize=10)
    plt.axis('off')

    plt.subplot(layout[0], layout[1], 6)
    plt.imshow(exy, cmap='jet', interpolation='nearest', norm=normexy)
    plt.colorbar()
    plt.title("exy", fontsize=10)
    plt.axis('off')

    # --- save ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, bbox_inches='tight')
    logger.info(f"Result figure saved to {file_path}")
    plt.close()
    
def result_uv_plot(u, v, 
                layout=[1,2], WH=[5,4],
                save_dir=None, filename=None):
    """
    layout:
    [ u   v ]
    """

    # --- min/max (ignore zero) ---
    u_min, u_max = _nonzero_minmax(u)
    v_min, v_max = _nonzero_minmax(v)

    # --- zero -> nan ---
    u = zero_to_nan(u)
    v = zero_to_nan(v)

    # --- figure ---
    plt.figure(figsize=(WH[1]*layout[1], WH[0]*layout[0]), dpi=200)

    normu   = matplotlib.colors.Normalize(vmin=u_min,   vmax=u_max)
    normv   = matplotlib.colors.Normalize(vmin=v_min,   vmax=v_max)

    # ===== Row 1: displacement =====
    plt.subplot(layout[0], layout[1], 1)
    plt.imshow(u, cmap='jet', interpolation='nearest', norm=normu)
    plt.colorbar()
    plt.title("u", fontsize=10)
    plt.axis('off')

    plt.subplot(layout[0], layout[1], 2)
    plt.imshow(v, cmap='jet', interpolation='nearest', norm=normv)
    plt.colorbar()
    plt.title("v", fontsize=10)
    plt.axis('off')

    # --- save ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, bbox_inches='tight')
    logger.info(f"Result figure saved to {file_path}")
    plt.close()
    
def result_uvw_plot(u, v, w,
                layout=[1,3], WH=[5,4],
                save_dir=None, filename=None):
    """
    layout:
    [ u   v ]
    """

    # --- min/max (ignore zero) ---
    u_min, u_max = _nonzero_minmax(u)
    v_min, v_max = _nonzero_minmax(v)
    w_min, w_max = _nonzero_minmax(w)

    # --- zero -> nan ---
    u = zero_to_nan(u)
    v = zero_to_nan(v)
    w = zero_to_nan(w)

    # --- figure ---
    plt.figure(figsize=(WH[1]*layout[1], WH[0]*layout[0]), dpi=200)

    normu   = matplotlib.colors.Normalize(vmin=u_min,   vmax=u_max)
    normv   = matplotlib.colors.Normalize(vmin=v_min,   vmax=v_max)
    normw   = matplotlib.colors.Normalize(vmin=w_min,   vmax=w_max)

    # ===== Row 1: displacement =====
    plt.subplot(layout[0], layout[1], 1)
    plt.imshow(u, cmap='jet', interpolation='nearest', norm=normu)
    plt.colorbar()
    plt.title("u", fontsize=10)
    plt.axis('off')

    plt.subplot(layout[0], layout[1], 2)
    plt.imshow(v, cmap='jet', interpolation='nearest', norm=normv)
    plt.colorbar()
    plt.title("v", fontsize=10)
    plt.axis('off')
    
    plt.subplot(layout[0], layout[1], 3)
    plt.imshow(w, cmap='jet', interpolation='nearest', norm=normw)
    plt.colorbar()
    plt.title("w", fontsize=10)
    plt.axis('off')

    # --- save ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, bbox_inches='tight')
    logger.info(f"Result figure saved to {file_path}")
    plt.close()
    
def result_xyz_plot(X, Y, Z,
                   WH=[5,4],
                   elev=30, azim=-60,
                   save_dir=None, filename=None):
    """
    3D surface visualization

    Parameters
    ----------
    X, Y, Z : 2D array
        coordinate matrices
    WH : figure size scaling
    elev, azim : view angle
    """

    # ===== 处理数据 =====
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)

    X = X.astype(float)
    Y = Y.astype(float)
    Z = Z.astype(float)

    # ===== figure =====
    fig = plt.figure(figsize=(WH[1], WH[0]), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # colormap
    norm = matplotlib.colors.Normalize(vmin=z_min, vmax=z_max)
    cmap = plt.get_cmap('jet')

    # surface
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=cmap(norm(Z)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False
    )

    # ✅ 更标准的 colorbar 写法
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])   # ← 这里建议改为空数组（更规范）

    fig.colorbar(mappable, ax=ax, shrink=0.6)

    # view
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("3D Surface", fontsize=10)

    # 去坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # ===== save =====
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Saved to {file_path}")


def result_3d_displacement_surface(X, Y, Z, U, V, W,
                                   WH=[5, 4],
                                   elev=30, azim=-60,
                                   save_dir=None, filename=None):
    """3D deformed surface colored by total displacement magnitude.

    Plots the deformed 3D surface (X, Y, Z) with face colors mapped to
    the total displacement sqrt(U²+V²+W²). NaN-safe.

    Parameters
    ----------
    X, Y, Z : 2D array
        Deformed 3D coordinates (from triangulation).
    U, V, W : 2D array
        3D displacement components.
    WH : list
        Figure size scaling [height, width].
    elev, azim : float
        View angle (elevation, azimuth).
    save_dir : str
        Output directory (e.g. fig_3d_dir).
    filename : str
        Output filename (e.g. "surface_frame_001.png").
    """
    # --- Compute total displacement magnitude ---
    U_flat = U.ravel()
    V_flat = V.ravel()
    W_flat = W.ravel()
    disp_mag = np.sqrt(U_flat**2 + V_flat**2 + W_flat**2)
    D = disp_mag.reshape(U.shape)

    d_min = np.nanmin(D)
    d_max = np.nanmax(D)
    if not np.isfinite(d_min):
        d_min, d_max = 0.0, 1.0

    Xf = X.astype(float)
    Yf = Y.astype(float)
    Zf = Z.astype(float)

    # --- Figure ---
    fig = plt.figure(figsize=(WH[1], WH[0]), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    norm = matplotlib.colors.Normalize(vmin=d_min, vmax=d_max)
    cmap = plt.get_cmap('jet')

    surf = ax.plot_surface(
        Xf, Yf, Zf,
        facecolors=cmap(norm(D)),
        rstride=1, cstride=1,
        linewidth=0, antialiased=False, shade=False,
    )

    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6)
    cbar.set_label("|Displacement| (mm)", fontsize=8)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title("3D Deformed Surface", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # --- Save ---
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight')
        logger.info(f"3D displacement surface saved to {file_path}")
        plt.close(fig)


def result_3d_deformation_plot(U, V, W, exx, eyy, ezz, exy, exz, eyz,
                                save_dir=None, filename=None):
    """3x3 grid plot of 3D deformation (displacement + strain).

    Layout:
        U    V    W
       exx  eyy  ezz
       exy  exz  eyz
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), dpi=150)

    titles = ["U", "V", "W", "exx", "eyy", "ezz", "exy", "exz", "eyz"]
    data = [U, V, W, exx, eyy, ezz, exy, exz, eyz]

    for ax, title, d in zip(axes.flat, titles, data):
        d_nan = np.where(np.abs(d) < 1e-30, np.nan, d)
        im = ax.imshow(d_nan, cmap='jet', interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("3D Deformation", fontsize=12, y=1.01)
    plt.tight_layout()

    if save_dir is not None and filename is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, filename)
        fig.savefig(file_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info(f"Deformation plot saved to {file_path}")
    elif save_dir is not None:
        return fig
    else:
        plt.close(fig)

    plt.close()