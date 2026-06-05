"""
PINN-Multi-DIC: Multi-camera 3D DIC pipeline.

Efficient two-stage approach:
  Stage 1 (ref only): Cross-view DIC + triangulation → 3D reference surface
  Stage 2 (per frame): Intra-view temporal DIC per camera → propagate
                       correspondences → triangulate → 3D displacement
"""

from pinndicmulti.DIC_importlib import os, jax, jnp, np, savemat

from pinndicmulti.DIC_config import (
    seed_config_txt, DIC_3D_config_txt, fusion_config_txt,
    discover_cameras, get_work_subdir,
)
from pinndicmulti.reconstruction.DIC_calibrate import colmap_calibrate_multi_camera
from pinndicmulti.reconstruction.DIC_triangulation import triangulate_all_pairs
from pinndicmulti.reconstruction.DIC_fusion_nn import FusionTrainer
from pinndicmulti.reconstruction.DIC_strain3Dcalc import DIC3D_Strain_from_Displacement
from pinndicmulti.segpinndic.DIC_readImg import BufferManager, MultiCamDataset
from pinndicmulti.segpinndic.DIC_seedcalc import CalcSeeds, Seed_match_visualization
from pinndicmulti.segpinndic.utils.logger import logger
from pinndicmulti.segpinndic.DIC_trainers import FBPINNTrainer, PINNTrainer
from pinndicmulti.segpinndic.DIC_constants import Constants
from pinndicmulti.segpinndic import DIC_networks
from pinndicmulti.segpinndic.DIC_plot_trainer import (
    result_uv_plot, result_uvw_plot, result_xyz_plot,
    result_3d_deformation_plot, result_3d_displacement_surface,
)


def _run_dic_on_pair(trainer, constants_list, DIC_config, N_roi):
    """Run 2D DIC on the current ref→def pair (BufferManager must be set up).

    Returns (u, v, exx, exy, eyy, x_batch_global) as numpy arrays.
    """
    u = v = exx = exy = eyy = jnp.zeros_like(BufferManager.refImg)

    for roi_id in range(N_roi):
        c = constants_list[roi_id]
        mu = np.array([
            BufferManager.scale_uv[roi_id][0],
            BufferManager.scale_uv[roi_id][1],
        ])
        sd = np.array([
            BufferManager.scale_uv[roi_id][2],
            BufferManager.scale_uv[roi_id][3],
        ])
        logger.info(f"roi_{roi_id} scale_uv: mu={mu}, sd={sd}  "
                    f"seed_pos={'None' if c.seed_pos is None else len(c.seed_pos)} "
                    f"seed_train_epochs={c.seed_train_epochs}  "
                    f"sd_threshold={c.seed_sd_threshold}")
        c.load_kwargs(
            decomposition_init_kwargs=dict(
                subdomain_xs=c.subdomain_xs,
                subdomain_ws=c.subdomain_ws,
                unnorm=(mu, sd),
            ),
            problem_init_kwargs=dict(
                ref_img=BufferManager.refImg,
                QKBQKT_def=BufferManager.QKBQKT_def_DIC,
                mask=BufferManager.mask[roi_id],
                degree=DIC_config.spline_degree,
            ),
        )
        run = trainer(c)
        _, u_, v_, exx_, exy_, eyy_, x_batch_global_ = run.train()

        if N_roi == 1:
            u, v, exx, exy, eyy = u_, v_, exx_, exy_, eyy_
        else:
            ys_b = x_batch_global_[:, 1].astype(jnp.int32)
            xs_b = x_batch_global_[:, 0].astype(jnp.int32)
            u = u.at[ys_b, xs_b].set(u_.flatten())
            v = v.at[ys_b, xs_b].set(v_.flatten())
            exx = exx.at[ys_b, xs_b].set(exx_.flatten())
            exy = exy.at[ys_b, xs_b].set(exy_.flatten())
            eyy = eyy.at[ys_b, xs_b].set(eyy_.flatten())

    return (np.asarray(u), np.asarray(v),
            np.asarray(exx), np.asarray(exy), np.asarray(eyy),
            x_batch_global_)


def _prepare_ref_and_def(dataset, cam_idx, frame_idx_or_path):
    """Set up BufferManager for DIC: load ref & def images, compute B-spline buffers.

    When frame_idx_or_path is an int, load from dataset.cam_def_files.
    When it's a string, use it directly as the image path (for cross-view).
    """
    from pinndicmulti.segpinndic.DIC_readImg import build_seed_buffer_jax, build_DIC_buffer_jax

    # Reference: this camera's reference image
    dataset.set_ref_image(cam_idx)

    # Deformed image
    if isinstance(frame_idx_or_path, int):
        def_path = dataset.cam_def_files[cam_idx][frame_idx_or_path]
    else:
        def_path = frame_idx_or_path

    BufferManager.defImg = MultiCamDataset.open_image(def_path)
    BufferManager.defImg_pad = jnp.pad(
        BufferManager.defImg,
        pad_width=dataset.coarse_subset_radius,
        mode='constant', constant_values=False,
    )

    # Ensure seed buffers exist (use this camera's ref + mask)
    mask_bin = np.asarray(jnp.logical_or.reduce(
        jnp.stack(BufferManager.mask, axis=0), axis=0))
    build_seed_buffer_jax(BufferManager.refImg, mask_bin, degree=5)
    build_DIC_buffer_jax(BufferManager.defImg, degree=dataset.spline_degree)


def _setup_seed_uv(DIC_config, Seed_config, N_roi):
    """Run seed matching and set BufferManager.scale_uv. Returns seed_pos, seed_uv."""
    SeedCalculator = CalcSeeds(Seed_config)
    if DIC_config.seed_flag:
        seed_pos, seed_uv = SeedCalculator.analyze()
        BufferManager.scale_uv = [
            jnp.asarray((
                (jnp.max(a[:, 0]) + jnp.min(a[:, 0])) / 2,
                (jnp.max(a[:, 1]) + jnp.min(a[:, 1])) / 2,
                (jnp.max(a[:, 0]) - jnp.min(a[:, 0])) / 2,
                (jnp.max(a[:, 1]) - jnp.min(a[:, 1])) / 2,
            )) for a in seed_uv
        ]
        return seed_pos, seed_uv
    else:
        BufferManager.scale_uv = [
            jnp.asarray((0., 0., 1., 1.)) for _ in range(N_roi)
        ]
        return None, None


def main(
    dic_config_path="./config/PINN-DIC-Mutil3D.txt",
    seed_config_path="./config/Seed_Configuration.txt",
    fusion_config_path="./config/Fusion_Configuration.txt",
):
    # ==============================================================
    # 1. Parse configuration
    # ==============================================================
    DIC_config = DIC_3D_config_txt(dic_config_path, verbose=False)
    Seed_config = seed_config_txt(seed_config_path, verbose=False)
    fusion_config = fusion_config_txt(fusion_config_path, verbose=False)
    work_dir = DIC_config.work_dir
    logger.info(f"Work directory: {work_dir}")

    # ==============================================================
    # 2. Discover cameras & load dataset
    # ==============================================================
    cam_dirs = discover_cameras(work_dir)
    num_cameras = len(cam_dirs)
    logger.info(f"Found {num_cameras} cameras: {cam_dirs}")

    mask_cam = getattr(DIC_config, 'mask_camera', "1")
    ImgData = MultiCamDataset(work_dir, Seed_config,
                              mask_path=DIC_config.mask_path,
                              mask_camera=mask_cam)
    ref_cam = ImgData.ref_cam_name
    logger.info(f"Reference camera (mask): {ref_cam}")
    num_frames = ImgData.num_frames
    N_roi = len(BufferManager.mask)
    mask_all = jnp.logical_or.reduce(jnp.stack(BufferManager.mask, axis=0), axis=0)
    ROInp = np.array(mask_all)
    ys_roi, xs_roi = np.where(ROInp)
    coords = np.stack([xs_roi, ys_roi], axis=1)  # (N, 2) in camera 0 ref frame
    img_h, img_w = int(mask_all.shape[0]), int(mask_all.shape[1])

    # ==============================================================
    # 3. COLMAP self-calibration
    # ==============================================================
    logger.info("Running COLMAP self-calibration...")
    calib = colmap_calibrate_multi_camera(work_dir)

    # Reorder calibration to match dataset ordering (mask_camera first).
    # calib uses sorted filesystem order; dataset puts mask_camera at index 0.
    calib_cam_names = []
    for n in calib["cam_names"].flat:
        if hasattr(n, 'flat') and not isinstance(n, str):
            calib_cam_names.append(str(n.flat[0]))
        else:
            calib_cam_names.append(str(n))
    ds_cam_names = ImgData.cam_names
    remap = [calib_cam_names.index(name) for name in ds_cam_names]
    logger.info(f"Calibration remap (fs→ds): {remap}")
    for key in ("K_list", "dist_list", "cam_from_world_R", "cam_from_world_t",
                "P_list", "camera_models", "camera_params_list", "cam_names"):
        calib[key] = [calib[key][i] for i in remap]

    # ==============================================================
    # 4. Initialize constants & output dirs
    # ==============================================================
    constants_list = []
    for roi_id in range(N_roi):
        c_ = Constants(DIC_config, roi_id)
        constants_list.append(c_)
        if roi_id == 0:
            c_.get_outdirs()
            c_.clear_outdirs()
            c_.save_constants_file()

    # Select trainer
    if np.prod(tuple(DIC_config.n_subdomains)) == 1:
        logger.info("Using PINN solver")
        trainer = PINNTrainer
    else:
        logger.info("Using FBPPINN solver")
        trainer = FBPINNTrainer

    # ==============================================================
    # STAGE 1: Cross-view DIC at reference frame (once)
    # ==============================================================
    # Cross-view: ref_cam ↔ other cameras → correspondence of every ROI pixel
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 1: Cross-view DIC (ref={ref_cam} ↔ others)")
    logger.info(f"{'='*60}")

    # Set up figure output subdirectories
    fig_3d_dir = os.path.join(c_.fig_out_dir, "3D")
    fig_2d_dir = os.path.join(c_.fig_out_dir, "2D")
    fig_disparity_dir = os.path.join(c_.fig_out_dir, "Disparity")

    ref_matched_pts = {0: coords.astype(np.float64)}  # ref camera: identity

    for j in range(1, num_cameras):
        logger.info(f"  Cross-view: cam0_ref ↔ cam{j}_ref")

        # Set up DIC: cam0 ref vs cam_j ref
        _prepare_ref_and_def(ImgData, 0, ImgData.cam_ref_files[j])
        seed_pos, seed_uv = _setup_seed_uv(DIC_config, Seed_config, N_roi)

        # Wire seed_pos/seed_uv into constants so trainers can use them.
        # Only enable if displacement half-range (sd) exceeds threshold.
        if seed_pos is not None:
            for roi_id, c_ in enumerate(constants_list):
                if roi_id < len(seed_pos):
                    sd_u = float(BufferManager.scale_uv[roi_id][2])
                    sd_v = float(BufferManager.scale_uv[roi_id][3])
                    if sd_u > c_.seed_sd_threshold or sd_v > c_.seed_sd_threshold:
                        c_.seed_pos = seed_pos[roi_id]
                        c_.seed_uv = seed_uv[roi_id]
                        logger.info(
                            f"roi_{roi_id} seed pre-training ENABLED: "
                            f"sd=({sd_u:.2f}, {sd_v:.2f}) > "
                            f"threshold={c_.seed_sd_threshold}")
                    else:
                        c_.seed_pos = None
                        c_.seed_uv = None
                        logger.info(
                            f"roi_{roi_id} seed pre-training SKIPPED: "
                            f"sd=({sd_u:.2f}, {sd_v:.2f}) <= "
                            f"threshold={c_.seed_sd_threshold}")

        if DIC_config.save_figures and seed_pos is not None:
            Seed_match_visualization(
                BufferManager.refImg * 255, BufferManager.defImg * 255,
                seed_pos, seed_uv, c_.fig_out_dir,
                f"crossview_0_{j}", j,
            )

        # Run DIC
        u_cv, v_cv, _, _, _, _ = _run_dic_on_pair(
            trainer, constants_list, DIC_config, N_roi)

        # Save cross-view disparity plot
        if DIC_config.save_figures:
            result_uv_plot(u_cv, v_cv, save_dir=fig_disparity_dir,
                          filename=f"crossview_0_{j}.png")

        # Matched points in camera j: coords + cross-view disparity
        u_cv_roi = u_cv[coords[:, 1], coords[:, 0]]
        v_cv_roi = v_cv[coords[:, 1], coords[:, 0]]
        ref_matched_pts[j] = coords.astype(np.float64) + \
            np.column_stack([u_cv_roi, v_cv_roi])

        logger.info(f"  cam0↔cam{j}: u [{np.min(u_cv_roi):.2f}, {np.max(u_cv_roi):.2f}], "
                    f"v [{np.min(v_cv_roi):.2f}, {np.max(v_cv_roi):.2f}]")

    # Triangulate → 3D reference surface
    logger.info("Triangulating reference surface...")
    _, pts3D_all_ref = triangulate_all_pairs(calib, ref_matched_pts, coords,
                                             return_raw=True)

    # Neural network fusion
    logger.info(f"Fusing reference surface ({len(pts3D_all_ref)} pairs, "
                f"{coords.shape[0]} points)...")
    fusion_ref = FusionTrainer(fusion_config)
    key_ref = jax.random.PRNGKey(42)
    pts3D_ref = fusion_ref.train(coords, pts3D_all_ref, key=key_ref)

    Xref = np.empty((img_h, img_w), dtype=np.float32); Xref.fill(np.nan)
    Yref = np.empty((img_h, img_w), dtype=np.float32); Yref.fill(np.nan)
    Zref = np.empty((img_h, img_w), dtype=np.float32); Zref.fill(np.nan)
    Xref[coords[:, 1], coords[:, 0]] = pts3D_ref[:, 0]
    Yref[coords[:, 1], coords[:, 0]] = pts3D_ref[:, 1]
    Zref[coords[:, 1], coords[:, 0]] = pts3D_ref[:, 2]

    # Save reference surface
    reconstruct_dir = c_.reconstruct_out_dir
    deformation_dir = c_.deformation_out_dir

    savemat(os.path.join(reconstruct_dir, "DIC_000.mat"),
            {"X": Xref, "Y": Yref, "Z": Zref})
    logger.info(f"Reference surface saved")

    if DIC_config.save_figures:
        result_xyz_plot(Xref, Yref, Zref,
                        save_dir=fig_3d_dir, filename="surface_ref.png")

    # ==============================================================
    # STAGE 2: Cross-temporal DIC per frame → triangulate → 3D
    # ==============================================================
    # All matching is anchored to cam3_ref (mask camera reference).
    # For each frame t and each camera j:
    #   DIC(cam3_ref, cam_j_def_t) → 亚像素匹配点
    #   Triangulate from all cameras → Xt, Yt, Zt
    #   U,V,W = Xt - Xref
    #
    # This guarantees TRUE corresponding points (same physical surface
    # point) because every DIC evaluates displacement at the same
    # cam3 ROI pixel coordinates, and the PINN returns sub-pixel values.
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 2: Cross-temporal DIC ({num_cameras} cameras × {num_frames} frames)")
    logger.info(f"{'='*60}")

    for frame_idx in range(num_frames):
        logger.info(f"\n--- Frame {frame_idx+1}/{num_frames} ---")
        frame_matched_pts = {}

        for cam_idx in range(num_cameras):
            # Prepare: cam3_ref ↔ cam_j_def_at_frame_t
            if cam_idx == 0:
                # Same-camera temporal: cam3_ref vs cam3_def_t
                _prepare_ref_and_def(ImgData, 0, frame_idx)
            else:
                # Cross-camera + temporal: cam3_ref vs cam_j_def_t
                _prepare_ref_and_def(ImgData, 0,
                                     ImgData.cam_def_files[cam_idx][frame_idx])

            seed_pos, seed_uv = _setup_seed_uv(DIC_config, Seed_config, N_roi)

            # Wire seed_pos/seed_uv into constants
            if seed_pos is not None:
                for roi_id, c_ in enumerate(constants_list):
                    if roi_id < len(seed_pos):
                        sd_u = float(BufferManager.scale_uv[roi_id][2])
                        sd_v = float(BufferManager.scale_uv[roi_id][3])
                        if sd_u > c_.seed_sd_threshold or sd_v > c_.seed_sd_threshold:
                            c_.seed_pos = seed_pos[roi_id]
                            c_.seed_uv = seed_uv[roi_id]
                            logger.info(
                                f"roi_{roi_id} seed pre-training ENABLED: "
                                f"sd=({sd_u:.2f}, {sd_v:.2f}) > "
                                f"threshold={c_.seed_sd_threshold}")
                        else:
                            c_.seed_pos = None
                            c_.seed_uv = None
                            logger.info(
                                f"roi_{roi_id} seed pre-training SKIPPED: "
                                f"sd=({sd_u:.2f}, {sd_v:.2f}) <= "
                                f"threshold={c_.seed_sd_threshold}")

            if DIC_config.save_figures and seed_pos is not None:
                Seed_match_visualization(
                    BufferManager.refImg * 255, BufferManager.defImg * 255,
                    seed_pos, seed_uv, c_.fig_out_dir,
                    f"cam{cam_idx}_frame{frame_idx+1:03d}", frame_idx + 1,
                )

            # Run DIC: cam3_ref → cam_j_def_t
            u, v, _, _, _, _ = _run_dic_on_pair(
                trainer, constants_list, DIC_config, N_roi)

            # Save disparity plot (2D DIC field)
            if DIC_config.save_figures:
                result_uv_plot(u, v, save_dir=fig_disparity_dir,
                              filename=f"cam{cam_idx}_frame{frame_idx+1:03d}.png")

            # Extract sub-pixel displacement at cam3 ROI coordinates
            u_roi = u[coords[:, 1], coords[:, 0]]
            v_roi = v[coords[:, 1], coords[:, 0]]
            frame_matched_pts[cam_idx] = (
                coords.astype(np.float64) +
                np.column_stack([u_roi, v_roi])
            )

            logger.info(
                f"  cam{cam_idx}: u [{np.min(u_roi):.2f}, {np.max(u_roi):.2f}], "
                f"v [{np.min(v_roi):.2f}, {np.max(v_roi):.2f}]")

        # --- Triangulate current frame from all cameras ---
        _, pts3D_all_t = triangulate_all_pairs(calib, frame_matched_pts, coords,
                                                return_raw=True)

        # Neural network fusion
        logger.info(f"  Fusing frame {frame_idx+1} "
                    f"({len(pts3D_all_t)} pairs, {coords.shape[0]} points)...")
        fusion_frame = FusionTrainer(fusion_config)
        key_t = jax.random.PRNGKey(43 + frame_idx)
        pts3D_t = fusion_frame.train(coords, pts3D_all_t, key=key_t)

        # --- 3D displacement ---
        U_roi = pts3D_t[:, 0] - pts3D_ref[:, 0]
        V_roi = pts3D_t[:, 1] - pts3D_ref[:, 1]
        W_roi = pts3D_t[:, 2] - pts3D_ref[:, 2]

        U = np.empty((img_h, img_w), dtype=np.float32); U.fill(np.nan)
        V = np.empty((img_h, img_w), dtype=np.float32); V.fill(np.nan)
        W = np.empty((img_h, img_w), dtype=np.float32); W.fill(np.nan)
        U[coords[:, 1], coords[:, 0]] = U_roi
        V[coords[:, 1], coords[:, 0]] = V_roi
        W[coords[:, 1], coords[:, 0]] = W_roi

        Xt = np.empty((img_h, img_w), dtype=np.float32); Xt.fill(np.nan)
        Yt = np.empty((img_h, img_w), dtype=np.float32); Yt.fill(np.nan)
        Zt = np.empty((img_h, img_w), dtype=np.float32); Zt.fill(np.nan)
        Xt[coords[:, 1], coords[:, 0]] = pts3D_t[:, 0]
        Yt[coords[:, 1], coords[:, 0]] = pts3D_t[:, 1]
        Zt[coords[:, 1], coords[:, 0]] = pts3D_t[:, 2]

        # --- 3D strain ---
        exx, eyy, ezz, exy, exz, eyz = DIC3D_Strain_from_Displacement(
            Xref, Yref, Zref, U, V, W,
            ROInp,
            SmoothLen=DIC_config.strain_window_len,
        )

        # --- Save ---
        savemat(os.path.join(deformation_dir, f"DEF_{frame_idx+1:03d}.mat"), {
            "X": Xt, "Y": Yt, "Z": Zt,
            "U": U, "V": V, "W": W,
            "exx": exx, "eyy": eyy, "ezz": ezz,
            "exy": exy, "exz": exz, "eyz": eyz,
        })
        savemat(os.path.join(reconstruct_dir, f"DIC_{frame_idx+1:03d}.mat"),
                {"X": Xt, "Y": Yt, "Z": Zt})

        logger.info(
            f"Frame {frame_idx+1}/{num_frames}: "
            f"U=[{np.nanmin(U_roi):.3f},{np.nanmax(U_roi):.3f}] "
            f"V=[{np.nanmin(V_roi):.3f},{np.nanmax(V_roi):.3f}] "
            f"W=[{np.nanmin(W_roi):.3f},{np.nanmax(W_roi):.3f}]"
        )

        if DIC_config.save_figures:
            # --- 3D surface plots ---
            result_3d_displacement_surface(
                Xt, Yt, Zt, U, V, W,
                save_dir=fig_3d_dir,
                filename=f"surface_frame_{frame_idx+1:03d}.png",
            )
            # --- 2D heatmap plots ---
            result_uvw_plot(U, V, W, save_dir=fig_2d_dir,
                            filename=f"displacement_{frame_idx+1:03d}.png")
            result_3d_deformation_plot(
                U, V, W, exx, eyy, ezz, exy, exz, eyz,
                save_dir=fig_2d_dir,
                filename=f"strain_{frame_idx+1:03d}.png",
            )

    # ==============================================================
    # Done
    # ==============================================================
    logger.info(f"\n{'='*60}")
    logger.info("PINN-Multi-DIC pipeline complete!")
    logger.info(f"  Cameras: {num_cameras}  Frames: {num_frames}")
    logger.info(f"  Calibration: {c_.calib_out_dir}")
    logger.info(f"  Reconstruction: {reconstruct_dir}")
    logger.info(f"  Deformation: {deformation_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
