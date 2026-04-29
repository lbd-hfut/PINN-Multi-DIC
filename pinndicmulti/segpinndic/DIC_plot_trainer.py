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
    arr_nz = arr[arr != 0]
    if arr_nz.size == 0:
        return None, None
    return np.min(arr_nz), np.max(arr_nz)


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

    # ===== helper =====
    def zero_to_nan(arr):
        arr = arr.astype(float)
        arr[arr == 0] = np.nan
        return arr

    def nonzero_minmax(arr):
        valid = arr[arr != 0]
        return valid.min(), valid.max()

    # ===== 处理数据 =====
    z_min, z_max = nonzero_minmax(Z)

    X = zero_to_nan(X)
    Y = zero_to_nan(Y)
    Z = zero_to_nan(Z)

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

    plt.close()