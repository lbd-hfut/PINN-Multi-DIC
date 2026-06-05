"""
Neural network-based multi-view fusion for PINN-Multi-DIC.

Fits a coordinate-based MLP f(x, y) → (X, Y, Z) that maps
normalized ROI pixel coordinates to 3D world coordinates,
fusing multiple pairwise triangulation results into a
smooth, outlier-robust surface.

Uses network architectures from DIC_networks.py with full
JAX JIT compilation support.
"""

from pinndicmulti.DIC_importlib import (
    time, jaxopt, jnp, np,
    jit, vmap, value_and_grad, partial, random, optax
)
from pinndicmulti.segpinndic.utils.logger import logger
from pinndicmulti.segpinndic.utils.jax_util import partition, combine, total_size
from pinndicmulti.segpinndic import DIC_networks


class FusionTrainer:
    """Neural network-based multi-view 3D surface fusion.

    Trains a coordinate MLP (from DIC_networks.py) to map
    ROI pixel coordinates → 3D world coordinates, using
    multiple pairwise triangulation results as training data.

    Parameters
    ----------
    fusion_config : SimpleNamespace
        Parsed fusion configuration.
    """

    def __init__(self, fusion_config):
        self.c = fusion_config
        self.network_cls = getattr(DIC_networks, self.c.network)

        # Populated during train()
        self._built = False
        self.all_params = None
        self._mode = None
        self.xy_mu = None
        self.xy_sd = None
        self.xyz_mu = None
        self.xyz_sd = None

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_network(self, key):
        """Initialize network parameters for single or triple output mode."""
        c = self.c
        in_dim = 2

        if c.output_mode == "single":
            layer_sizes = [in_dim] + [c.hidden_neurons] * c.hidden_layers + [3]
            static_ps, trainable_ps = self.network_cls.init_params(key, layer_sizes)
            self.all_params = {
                "static": {"network": static_ps},
                "trainable": {"network": {"subdomain": trainable_ps}},
            }
            self._mode = "single"

        elif c.output_mode == "triple":
            layer_sizes = [in_dim] + [c.hidden_neurons] * c.hidden_layers + [1]
            key_X, key_Y, key_Z = random.split(key, 3)

            def _init_one(k):
                s, t = self.network_cls.init_params(k, layer_sizes)
                return s, {"subdomain": t}

            sX, tX = _init_one(key_X)
            sY, tY = _init_one(key_Y)
            sZ, tZ = _init_one(key_Z)

            self.all_params = {
                "static": {"network_X": sX, "network_Y": sY, "network_Z": sZ},
                "trainable": {
                    "network_X": tX,
                    "network_Y": tY,
                    "network_Z": tZ,
                },
            }
            self._mode = "triple"
        else:
            raise ValueError(f"Unknown output_mode: {c.output_mode}")

        self._built = True

    # ------------------------------------------------------------------
    # Data normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_xy(roi_coords):
        """Normalize ROI pixel coords to approximately [-1, 1]²."""
        xy_mu = roi_coords.mean(axis=0)
        xy_sd = (roi_coords.max(axis=0) - roi_coords.min(axis=0)) / 2.0
        xy_sd = jnp.maximum(xy_sd, 1.0)
        return xy_mu, xy_sd

    @staticmethod
    def _normalize_xyz(xyz):
        """Compute mu, sd for XYZ standardization."""
        xyz_mu = xyz.mean(axis=0)
        xyz_sd = xyz.std(axis=0)
        xyz_sd = jnp.maximum(xyz_sd, 1e-8)
        return xyz_mu, xyz_sd

    # ------------------------------------------------------------------
    # Training data construction
    # ------------------------------------------------------------------

    def _build_training_data(self, roi_coords, pairwise_pts3D):
        """Build normalized training data with optional outlier weights.

        Always returns fixed-shape arrays (M*N, 2) and (M*N, 3) with
        per-sample weights for JIT compatibility. Outlier samples get
        weight=0 so they don't contribute to the loss.

        Returns
        -------
        x_train : jnp.ndarray (M*N, 2)
        y_train_raw : jnp.ndarray (M*N, 3)
        weights : jnp.ndarray (M*N,)
        xy_mu, xy_sd, xyz_mu, xyz_sd
        """
        stacked = np.stack(pairwise_pts3D, axis=0)  # (M, N, 3)
        M, N, _ = stacked.shape
        total_samples = M * N

        # ROI coordinate normalization
        xy_mu, xy_sd = self._normalize_xy(roi_coords)
        x_norm = (roi_coords - xy_mu) / xy_sd

        # Repeat x_norm for each pair → (M*N, 2)
        x_train = np.tile(x_norm, (M, 1))
        y_train_raw = stacked.reshape(-1, 3)

        # --- NaN / inf mask ---
        nan_mask = ~np.any(~np.isfinite(y_train_raw), axis=1)  # (M*N,)

        # XYZ normalization (use only valid/finite values)
        valid_flat = y_train_raw[nan_mask]
        if len(valid_flat) == 0:
            raise RuntimeError("All triangulated points are NaN/inf — "
                               "check calibration and DIC results.")
        xyz_mu, xyz_sd = self._normalize_xyz(valid_flat)

        # Build per-sample weights
        if self.c.prefilter_outliers:
            # Per-point median across pairs (NaN-safe)
            median_xyz = np.nanmedian(stacked, axis=0)  # (N, 3)
            diff = stacked - median_xyz[np.newaxis, :, :]  # (M, N, 3)
            dist = np.linalg.norm(diff, axis=2)  # (M, N)

            global_std = np.nanstd(dist)
            threshold = max(self.c.outlier_threshold_sigma * global_std, 1e-6)

            inlier_mask = (dist <= threshold).reshape(-1)  # (M*N,)
            kept = int(np.sum(inlier_mask & nan_mask))
            rejected = total_samples - kept

            logger.info(
                f"  Prefilter: {kept}/{total_samples} samples kept "
                f"({rejected} outliers/NaN rejected, "
                f"sigma={global_std:.4f} mm, threshold={threshold:.4f} mm)"
            )
            weights = (inlier_mask & nan_mask).astype(np.float32)
        else:
            n_nan = total_samples - int(np.sum(nan_mask))
            logger.info(f"  Training data: {total_samples} samples from {M} pairs "
                        f"({n_nan} NaN/inf excluded)")
            weights = nan_mask.astype(np.float32)

        return (
            jnp.asarray(x_train),
            jnp.asarray(y_train_raw),
            jnp.asarray(weights),
            xy_mu, xy_sd, xyz_mu, xyz_sd,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, roi_coords, pairwise_pts3D, key=None):
        """Train the fusion network on multi-view triangulation data.

        Parameters
        ----------
        roi_coords : ndarray (N, 2)
            ROI pixel coordinates (x, y).
        pairwise_pts3D : list of ndarray
            M groups of (N, 3) triangulated 3D point clouds.
        key : jax.random.PRNGKey, optional

        Returns
        -------
        pts3D_fused : ndarray (N, 3)
            Fused 3D point cloud.
        """
        c = self.c
        if key is None:
            key = random.PRNGKey(0)

        # ---- Build training data ----
        (x_train, y_train_raw, weights,
         xy_mu, xy_sd, xyz_mu, xyz_sd) = self._build_training_data(
            roi_coords, pairwise_pts3D)

        self.xy_mu = xy_mu
        self.xy_sd = xy_sd
        self.xyz_mu = xyz_mu
        self.xyz_sd = xyz_sd

        logger.info(f"  xy_mu={np.array2string(np.asarray(xy_mu), precision=2)}, "
                    f"xy_sd={np.array2string(np.asarray(xy_sd), precision=2)}")
        logger.info(f"  xyz_mu={np.array2string(np.asarray(xyz_mu), precision=4)}, "
                    f"xyz_sd={np.array2string(np.asarray(xyz_sd), precision=4)}")

        # Normalize targets
        y_norm = (y_train_raw - xyz_mu) / xyz_sd

        # Replace NaN/inf with 0.0 (masked by weights=0 in loss, avoids NaN*0=NaN)
        y_norm = jnp.nan_to_num(y_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- Build network ----
        key_net, key_train = random.split(key)
        self._build_network(key_net)

        n_params = total_size(self.all_params["trainable"])
        logger.info(f"  Network: {c.network}, mode={self._mode}, "
                    f"hidden=[{c.hidden_neurons}]*{c.hidden_layers}, "
                    f"params={n_params:,}")

        # ---- Setup optimizer ----
        optimiser = optax.adam(c.adam_lr)
        active_params = self.all_params["trainable"]
        static_params = self.all_params["static"]
        active_opt_states = optimiser.init(active_params)

        # AOT compile update step
        n_steps = c.adam_epochs
        logger.info(f"  Compiling update step ({n_steps} Adam epochs)...")

        static_dynamic, static_static = partition(static_params)
        network_fn = self.network_cls.network_fn

        update = _fusion_update.lower(
            optimiser.update, active_opt_states,
            active_params, static_dynamic, static_static,
            network_fn, self._mode,
            x_train, y_norm, weights,
        ).compile()

        logger.info("  Compilation done")

        # ---- Adam training ----
        start_time = time.time()

        for i in range(n_steps):
            lossval, active_opt_states, active_params = update(
                active_opt_states, active_params, static_dynamic,
                x_train, y_norm, weights,
            )

            if (i + 1) % c.summary_freq == 0 or i == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"  [Adam {i+1}/{n_steps}] loss: {lossval.item():.6e} | "
                    f"{elapsed:.1f}s"
                )

        self.all_params["trainable"] = active_params

        # ---- L-BFGS refinement ----
        if c.lbfgs_epochs > 0:
            logger.info(
                f"  Starting L-BFGS refinement ({c.lbfgs_epochs} steps)..."
            )

            solver = jaxopt.LBFGS(
                fun=lambda ap, sp_dyn, xb, yb, w: _fusion_loss(
                    ap, combine(sp_dyn, static_static),
                    xb, yb, w, network_fn, self._mode,
                ),
                maxiter=1,
                tol=c.lbfgs_tol,
                maxls=c.lbfgs_maxls,
                history_size=c.lbfgs_history_size,
                stepsize=0.0,
                max_stepsize=c.lbfgs_lr if c.lbfgs_lr > 0.0 else 1.0,
                implicit_diff=False,
            )

            lbfgs_state = solver.init_state(
                active_params, static_dynamic, x_train, y_norm, weights,
            )

            loss_init = lbfgs_state.value
            logger.info(
                f"  [L-BFGS: 0/{c.lbfgs_epochs}] loss: {loss_init.item():.6e}"
            )

            start_lbfgs = time.time()
            for j in range(c.lbfgs_epochs):
                active_params, lbfgs_state = solver.update(
                    active_params, lbfgs_state,
                    static_dynamic, x_train, y_norm, weights,
                )
                if (j + 1) % c.summary_freq == 0 or j == 0:
                    loss_l = lbfgs_state.value
                    logger.info(
                        f"  [L-BFGS: {j+1}/{c.lbfgs_epochs}] "
                        f"loss: {loss_l.item():.6e} | "
                        f"{time.time() - start_lbfgs:.1f}s"
                    )

            self.all_params["trainable"] = active_params
            logger.info(f"  L-BFGS done ({time.time() - start_lbfgs:.1f}s)")

        # ---- Predict fused surface ----
        pts3D_fused = self.predict(roi_coords)

        return pts3D_fused

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, roi_coords):
        """Predict fused 3D points for given ROI coordinates.

        Parameters
        ----------
        roi_coords : ndarray (N, 2)
            ROI pixel coordinates.

        Returns
        -------
        pts3D : ndarray (N, 3)
            Fused 3D world coordinates.
        """
        if not self._built:
            raise RuntimeError("Network not built. Call train() first.")

        x_norm = jnp.asarray((roi_coords - self.xy_mu) / self.xy_sd)
        network_fn = self.network_cls.network_fn

        # Batch prediction for large point sets
        N = x_norm.shape[0]
        batch_size = 512 * 512

        if N > batch_size:
            preds = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                x_b = x_norm[start:end]
                preds.append(
                    _fusion_forward(self.all_params, x_b, network_fn, self._mode)
                )
            pred_norm = jnp.concatenate(preds, axis=0)
        else:
            pred_norm = _fusion_forward(
                self.all_params, x_norm, network_fn, self._mode
            )

        # Denormalize
        pts3D = np.asarray(pred_norm * self.xyz_sd + self.xyz_mu)
        return pts3D


# ====================================================================
# JIT-compiled functions (module level for picklability)
# ====================================================================

def _fusion_forward(params, x, network_fn, mode):
    """Forward pass through the fusion network(s).

    Parameters
    ----------
    params : dict
        Full params dict {"static": ..., "trainable": ...}.
    x : ndarray (N, 2)
        Normalized ROI coordinates.
    network_fn : callable
        Static network forward function (e.g. AdaptiveFCN.network_fn).
    mode : str
        "single" or "triple".

    Returns
    -------
    pred : ndarray (N, 3)
    """
    if mode == "single":
        return vmap(network_fn, in_axes=(None, 0))(params, x)
    else:
        def _forward_component(comp_params, x_batch):
            return vmap(network_fn, in_axes=(None, 0))(comp_params, x_batch)

        def _make_params(coord):
            # network_fn expects params["trainable"]["network"]["subdomain"]["layers"]
            # but triple mode stores under "network_X/Y/Z" — remap to "network"
            return {
                "static": params["static"][f"network_{coord}"],
                "trainable": {"network": params["trainable"][f"network_{coord}"]},
            }

        pred_X = _forward_component(_make_params("X"), x)
        pred_Y = _forward_component(_make_params("Y"), x)
        pred_Z = _forward_component(_make_params("Z"), x)
        return jnp.concatenate([pred_X, pred_Y, pred_Z], axis=1)


def _fusion_loss(active_params, static_params, x_batch, y_batch, weights,
                 network_fn, mode):
    """Weighted MSE loss for fusion training."""
    all_params = {"static": static_params, "trainable": active_params}
    pred = _fusion_forward(all_params, x_batch, network_fn, mode)
    sq_errors = jnp.sum((pred - y_batch) ** 2, axis=1)  # (N,)
    weighted = sq_errors * weights
    return jnp.sum(weighted) / (jnp.sum(weights) + 1e-8)


@partial(jit, static_argnums=(0, 4, 5, 6))
def _fusion_update(optimiser_fn, active_opt_states,
                   active_params, static_params_dynamic, static_params_static,
                   network_fn, mode,
                   x_batch, y_batch, weights):
    """Single Adam update step — JIT compiled."""
    static_params = combine(static_params_dynamic, static_params_static)
    lossval, grads = value_and_grad(_fusion_loss, argnums=0)(
        active_params, static_params,
        x_batch, y_batch, weights,
        network_fn, mode,
    )
    updates, active_opt_states = optimiser_fn(
        grads, active_opt_states, active_params
    )
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params
