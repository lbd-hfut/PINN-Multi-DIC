from pinndicmulti.DIC_importlib import os, pickle, jax, jnp, np, SummaryWriter, savemat

from pinndicmulti.DIC_config import seed_config_txt, DIC_2D_config_txt
from pinndicmulti.segpinndic.DIC_readImg import BufferManager, ImgDataset2D
from pinndicmulti.segpinndic.DIC_seedcalc import CalcSeeds, Seed_match_visualization

from pinndicmulti.segpinndic.utils.logger import logger
from pinndicmulti.segpinndic.utils.other import DictToObj
from pinndicmulti.segpinndic.DIC_trainers import FBPINNTrainer, PINNTrainer
from pinndicmulti.segpinndic.DIC_constants import Constants
from pinndicmulti.segpinndic import DIC_networks
from pinndicmulti.segpinndic.DIC_plot_trainer import result_uv_strain_plot

def main(
    seed_config_path="./config/Seed_Configuration.txt",
    dic_config_path="./config/PINN-DIC-2D.txt"
    ):
    
    DIC_config = DIC_2D_config_txt(dic_config_path, verbose=False)
    Seed_config = seed_config_txt(seed_config_path, verbose=False)
    
    ImgData = ImgDataset2D(DIC_config, Seed_config)
    SeedCalculator = CalcSeeds(Seed_config)
    
    N_pairs, N_roi = len(ImgData), len(BufferManager.mask)
    constants_list = []
    for roi_id in range(N_roi):
        c_ = Constants(DIC_config, roi_id)
        constants_list.append(c_)
        if roi_id < 1:
            # clear directories
            c_.get_outdirs(dim=2)
            c_.clear_outdirs(dim=2)
            c_.save_constants_file()
    
    if np.prod(tuple(DIC_config.n_subdomains)) == 1:
        logger.info("using PINN solver")
        trainer = PINNTrainer
    else:
        logger.info("using FBPPINN solver")
        trainer = FBPINNTrainer
    
    for i in range(N_pairs):
        ImgData.get_image(i)
        if DIC_config.seed_flag:
            seed_pos, seed_uv = SeedCalculator.analyze()
            if DIC_config.save_figures:
                Seed_match_visualization(
                    BufferManager.refImg*255, 
                    BufferManager.defImg*255,
                    seed_pos, seed_uv, DIC_config.output_dir, f'seed{i+1:03d}', i+1
                )
            BufferManager.scale_uv = [jnp.asarray((
                (jnp.max(a[:,0]) + jnp.min(a[:,0]))/2,
                (jnp.max(a[:,1]) + jnp.min(a[:,1]))/2,
                (jnp.max(a[:,0]) - jnp.min(a[:,0]))/2,
                (jnp.max(a[:,1]) - jnp.min(a[:,1]))/2)) for a in seed_uv]
            for roi_id in range(N_roi):
                logger.info(f"ROI_id{roi_id+1}: umax: {jnp.max(seed_uv[roi_id][:,0])}, v_max: {jnp.max(seed_uv[roi_id][:,1])}")
                logger.info(f"ROI_id{roi_id+1}: umin: {jnp.min(seed_uv[roi_id][:,0])}, v_min: {jnp.min(seed_uv[roi_id][:,1])}")
        else:
            BufferManager.scale_uv = [jnp.asarray((0.,0.,1.,1.)) for roi_id in range(N_roi)]
        u = v = exx = exy = eyy = jnp.zeros_like(BufferManager.refImg)
        for roi_id in range(N_roi):
            logger.info(f"Processing imgage pair {i+1}/{N_pairs} ROI {roi_id+1}/{N_roi}")
            c = constants_list[roi_id]
            mu = np.array([
                BufferManager.scale_uv[roi_id][0],
                BufferManager.scale_uv[roi_id][1]
            ])
            sd = np.array([
                BufferManager.scale_uv[roi_id][2],
                BufferManager.scale_uv[roi_id][3]
            ])
            c.load_kwargs(
                decomposition_init_kwargs=dict(
                    subdomain_xs=c.subdomain_xs,
                    subdomain_ws=c.subdomain_ws,
                    unnorm=(mu, sd)),
                problem_init_kwargs=dict(
                    ref_img = BufferManager.refImg,
                    QKBQKT_def = BufferManager.QKBQKT_def_DIC,
                    mask = BufferManager.mask[roi_id],
                    degree = DIC_config.spline_degree)
            )
            run = trainer(c)
            _, u_, v_, exx_, exy_, eyy_, x_batch_global_ = run.train()
            if N_roi == 1:
                u, v, exx, exy, eyy = u_, v_, exx_, exy_, eyy_
            else:
                ys = x_batch_global_[:, 1].astype(jnp.int32)
                xs = x_batch_global_[:, 0].astype(jnp.int32)
                u = u.at[ys, xs].set(u_.flatten())
                v = v.at[ys, xs].set(v_.flatten())
                exx = exx.at[ys, xs].set(exx_.flatten())
                exy = exy.at[ys, xs].set(exy_.flatten())
                eyy = eyy.at[ys, xs].set(eyy_.flatten())
        u, v, exx, exy, eyy = np.asarray(u), np.asarray(v), \
            np.asarray(exx), np.asarray(exy), np.asarray(eyy)
        save_mat_dir = c.mat_out_dir
        save_mat_path = os.path.join(save_mat_dir, f"segpinndic_{i+1:03d}.mat")
        savemat(
            save_mat_path,
            {
                "u": u, "v": v, "exx": exx, "exy": exy, "eyy": eyy,
            }
        )
        if DIC_config.save_figures:
            save_fig_dir = c.fig_out_dir
            result_uv_strain_plot(
                u, v, exx, exy, eyy,
                save_dir=save_fig_dir, filename=f"result_{i+1:03d}.png"
            )
            
            
if __name__ == "__main__":
    main()