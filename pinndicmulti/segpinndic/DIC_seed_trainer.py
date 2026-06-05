"""
Seed point supervised pre-training for PINN-DIC.

Trains the neural network to fit IC-GN seed point displacements before
photometric DIC optimization. This provides a physics-informed initialization
that helps convergence in large-deformation scenarios where boundary warp
gradients would otherwise destabilize global optimization.

PINN:  MSE 在 unnormalized network output (像素位移) 与真实种子点位移之间。
FBPINN: 每个子域独立拟合其区域内的种子点，loss 在 unnormalized output (u) 上计算。
        各子域使用各自的 unnorm 参数将 raw output 恢复为像素位移后比较。
"""

from pinndicmulti.DIC_importlib import time, jax, jnp, np, jit, vmap, value_and_grad, partial, optax
from pinndicmulti.segpinndic.utils.logger import logger
from pinndicmulti.segpinndic.DIC_trainers import (
    PINN_model_inner,
    FBPINN_model,
    get_inputs,
    tree_map_dicts,
)
from pinndicmulti.segpinndic.utils.jax_util import partition, combine


# ==============================================================================
# PINN Seed Training
# ==============================================================================

def PINN_seed_loss(active_params, static_params, seed_x, seed_uv, model_fns,
                   smooth_x=None, smooth_lambda=0.0):
    """MSE + L2 gradient-norm smoothness penalty."""
    norm_fn, network_fn, unnorm_fn = model_fns
    all_params = {"static": static_params, "trainable": active_params}
    u, _ = vmap(PINN_model_inner, in_axes=(None, 0, None, None, None))(
        all_params, seed_x, norm_fn, network_fn, unnorm_fn)
    loss = jnp.mean((u - seed_uv) ** 2)

    # smooth_x.shape[0] is concrete at trace time -> safe for Python if
    if smooth_x.shape[0] > 0:
        def u_at_x(x):
            u_out, _ = PINN_model_inner(all_params, x, norm_fn, network_fn, unnorm_fn)
            return u_out
        jac = vmap(jax.jacrev(u_at_x))(smooth_x)          # (M, 2, 2)
        grad_sq = jnp.sum(jac ** 2, axis=(1, 2))          # (M,)  sum over [u,v]x[dx,dy]
        loss = loss + smooth_lambda * jnp.mean(grad_sq)

    return loss


@partial(jit, static_argnums=(0, 4, 7))
def PINN_seed_update(optimiser_fn, active_opt_states,
                     active_params, static_params_dynamic, static_params_static,
                     seed_x, seed_uv, model_fns,
                     smooth_x=None, smooth_lambda=0.0):
    static_params = combine(static_params_dynamic, static_params_static)
    lossval, grads = value_and_grad(PINN_seed_loss, argnums=0)(
        active_params, static_params, seed_x, seed_uv, model_fns,
        smooth_x, smooth_lambda)
    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params


def train_seeds_pinn(all_params, seed_x, seed_uv, model_fns,
                     n_steps, learning_rate, summary_freq,
                     smooth_lambda=0.0, smooth_npoints=0, key=jax.random.PRNGKey(0)):
    """PINN seed supervised pre-training with optional smoothness regularisation.

    Args:
        all_params: model param dict {"static": ..., "trainable": ...}
        seed_x:   (N, 2) seed point pixel coordinates
        seed_uv:  (N, 2) seed point displacements (pixels)
        model_fns: (norm_fn, network_fn, unnorm_fn)
        n_steps:  Adam training steps
        learning_rate: learning rate
        summary_freq: log frequency
        smooth_lambda:  gradient-norm penalty weight (0 = disabled)
        smooth_npoints: number of ROI collocation points for smoothness (0 = disabled)
        key:    JAX PRNG key for smooth point sampling

    Returns:
        all_params with updated trainable params.
    """
    if seed_x is None or seed_uv is None or n_steps <= 0:
        return all_params

    # Generate smoothness collocation points (ROI interior)
    if smooth_lambda > 0 and smooth_npoints > 0:
        mask = all_params["static"]["problem"]["mask"]
        ys, xs = jnp.where(mask)
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, len(xs), (smooth_npoints,), replace=True)
        smooth_x = jnp.stack(
            [xs[idx].astype(jnp.float32), ys[idx].astype(jnp.float32)], axis=-1)
        logger.info(f"[Seed-PINN] Smoothness: lambda={smooth_lambda}, "
                    f"{smooth_npoints} collocation points")
    else:
        smooth_x = jnp.zeros((0, 2), dtype=jnp.float32)  # empty array for jit
        smooth_lambda = 0.0

    # Independent optimizer for seed training
    optimiser = optax.adam(learning_rate)
    active_params = all_params["trainable"]
    opt_states = optimiser.init(active_params)
    optimiser_fn = optimiser.update

    static_params = all_params["static"]
    static_params_dynamic, static_params_static = partition(static_params)

    # AOT compile
    logger.info("[Seed-PINN] Compiling seed training step...")
    t0 = time.time()
    update = PINN_seed_update.lower(
        optimiser_fn, opt_states,
        active_params, static_params_dynamic, static_params_static,
        seed_x, seed_uv, model_fns,
        smooth_x, smooth_lambda
    ).compile()
    logger.info(f"[Seed-PINN] Compilation done ({time.time() - t0:.2f}s)")

    # Training loop
    logger.info(f"[Seed-PINN] Seed training {n_steps} steps, "
                f"{seed_x.shape[0]} seed points...")
    t0 = time.time()
    for i in range(n_steps):
        lossval, opt_states, active_params = update(
            opt_states, active_params, static_params_dynamic,
            seed_x, seed_uv, smooth_x, smooth_lambda
        )
        if (i + 1) % summary_freq == 0 or i == 0:
            elapsed = time.time() - t0
            rate = min(i + 1, summary_freq) / elapsed if elapsed > 0 else float('inf')
            logger.info(f"[Seed-PINN {i+1}/{n_steps}] loss: {lossval.item():.6f}  "
                        f"rate: {rate:.1f} Hz")
            t0 = time.time()

    all_params["trainable"] = active_params
    logger.info(f"[Seed-PINN] Complete. Final loss: {lossval.item():.6f}")
    return all_params


# ==============================================================================
# FBPINN Seed Training
# ==============================================================================

def FBPINN_seed_loss(active_params, fixed_params, static_params, takes, x_batch,
                     model_fns, seed_uv):
    """每个子域的 unnormalized output 与真实种子点位移之间的 MSE。

    与 photometric loss 的关键区别：
    - 使用 FBPINN_model 的第6个返回值 us_u（unnormalized、未加窗的输出）
    - 直接在像素位移空间比较
    - 每个 (point, subdomain) pair 独立计算误差
    """
    # 合并 active + fixed params -> 完整的 trainable_params
    d, da = active_params, fixed_params
    trainable_params = {
        cl_k: {
            k: jax.tree_util.tree_map(
                lambda p1, p2: jnp.concatenate([p1, p2], 0),
                d[cl_k][k], da[cl_k][k]
            ) if k == "subdomain" else d[cl_k][k]
            for k in d[cl_k]
        }
        for cl_k in d
    }
    all_params = {"static": static_params, "trainable": trainable_params}

    # FBPINN_model 返回 (u, wp, us, ws, us_raw, us_u)
    # us_u: (s, ud) — 每个 (point, subdomain) pair 的 unnormalized 输出 (像素位移)
    _, _, _, _, _, us_u = FBPINN_model(all_params, x_batch, takes, model_fns)

    # n_take 将 pair index 映射到 point index
    _, n_take, _, _, _ = takes
    target = seed_uv[n_take]
    return jnp.mean((us_u - target) ** 2)


@partial(jit, static_argnums=(0, 5, 8))
def FBPINN_seed_update(optimiser_fn, active_opt_states,
                       active_params, fixed_params, static_params_dynamic,
                       static_params_static,
                       takes, x_batch, model_fns, seed_uv):
    static_params = combine(static_params_dynamic, static_params_static)
    lossval, grads = value_and_grad(FBPINN_seed_loss, argnums=0)(
        active_params, fixed_params, static_params, takes, x_batch,
        model_fns, seed_uv)
    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params


def train_seeds_fbpinn(all_params, seed_x, seed_uv, model_fns,
                       decomposition, n_steps, learning_rate, summary_freq):
    """FBPINN 种子点监督预训练。

    与 PINN 版本的关键区别：
    - 种子点通过 decomposition.inside_points 路由到各子域
    - 每个子域的 network 独立预测，各子域使用自身的 unnorm 参数将 raw output
      恢复为像素位移后与真实种子位移比较
    - 只有包含种子点的子域参与训练（其他保持随机初始化）
    - 不涉及 window/POU 组合——每个子域独立学习

    Args:
        all_params: 已初始化的模型参数字典
        seed_x: (N, 2) 种子点像素坐标
        seed_uv: (N, 2) 种子点位移 (像素)
        model_fns: (norm_fn, network_fn, unnorm_fn, window_fn)
        decomposition: Decomposition 类
        n_steps: Adam 训练步数
        learning_rate: 学习率
        summary_freq: 日志输出频率

    Returns:
        all_params with updated trainable params.
    """
    if seed_x is None or seed_uv is None or n_steps <= 0:
        return all_params

    # 所有子域设为 active
    m = all_params["static"]["decomposition"]["m"]
    active = jnp.ones(m, dtype=int)

    # 通过 get_inputs 确定每个种子点属于哪些子域
    takes, all_ims, (active, cut_active, cut_fixed, cut_all, merge_active) = \
        get_inputs(seed_x, active, all_params, decomposition)

    if len(all_ims) == 0:
        logger.warning("[Seed-FBPINN] No seed points in any subdomain, skipping.")
        return all_params

    # 裁剪参数到活跃子域
    active_params = cut_active(all_params["trainable"])
    fixed_params = cut_fixed(all_params["trainable"])
    static_params = cut_all(all_params["static"])

    # 为种子训练创建独立 optimizer
    optimiser = optax.adam(learning_rate)
    all_opt_states_seed = optimiser.init(all_params["trainable"])
    active_opt_states = tree_map_dicts(cut_active, all_opt_states_seed)
    optimiser_fn = optimiser.update

    static_params_dynamic, static_params_static = partition(static_params)

    n_active = len(jnp.unique(takes[0]))
    logger.info(f"[Seed-FBPINN] Compiling seed training step "
                f"({n_active}/{m} subdomains with seed points)...")
    t0 = time.time()
    update = FBPINN_seed_update.lower(
        optimiser_fn, active_opt_states,
        active_params, fixed_params, static_params_dynamic, static_params_static,
        takes, seed_x, model_fns, seed_uv
    ).compile()
    logger.info(f"[Seed-FBPINN] Compilation done ({time.time() - t0:.2f}s)")

    # 训练循环
    logger.info(f"[Seed-FBPINN] Seed training {n_steps} steps, "
                f"{seed_x.shape[0]} seed points...")
    t0 = time.time()
    for i in range(n_steps):
        lossval, active_opt_states, active_params = update(
            active_opt_states, active_params, fixed_params, static_params_dynamic,
            takes, seed_x, seed_uv
        )
        if (i + 1) % summary_freq == 0 or i == 0:
            elapsed = time.time() - t0
            rate = min(i + 1, summary_freq) / elapsed if elapsed > 0 else float('inf')
            logger.info(f"[Seed-FBPINN {i+1}/{n_steps}] loss: {lossval.item():.6f}  "
                        f"rate: {rate:.1f} Hz  subdomains: {n_active}")
            t0 = time.time()

    # 将训练好的子域参数合并回 all_params
    all_params["trainable"] = merge_active(active_params, all_params["trainable"])
    logger.info(f"[Seed-FBPINN] Complete. Final loss: {lossval.item():.6f}")
    return all_params
