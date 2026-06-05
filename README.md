<p align="center">
  <h1>PINN-Multi-DIC</h1>
  <p align="center">
    <em>Multi-Camera Physics-Informed Neural Network Framework for 3D Digital Image Correlation</em>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python version"></a>
    <a href="https://github.com/lbd-hfut/PINN-Multi-DIC/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
    <a href="https://github.com/lbd-hfut/PINN-Multi-DIC"><img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Development Status"></a>
  </p>
</p>

---

**PINN-Multi-DIC** extends PINN-based digital image correlation to **N-camera 3D measurement**. It uses COLMAP for automatic self-calibration and a three-stage efficient pipeline: cross-view DIC at the reference frame, intra-view temporal DIC per camera, and correspondence propagation for 3D deformation recovery — eliminating the need for calibration targets and stitching.

---

## Pipeline

```
 work_dir/images/
 ├── cam1/001.jpg ──┐
 ├── cam2/001.jpg ──┤──→ COLMAP SfM ──→ Camera params (K, dist, R, t)
 └── camN/001.jpg ──┘

 ═══════════════════════════════════════════════════
 STAGE 1: Cross-view DIC (reference frame only, once)
 ═══════════════════════════════════════════════════
         cam0_ref ←→ cam1_ref  跨视角 DIC ──→ 视差
         cam0_ref ←→ cam2_ref  跨视角 DIC ──→ 视差
                        ↓
               Pairwise 三角化 (M 组) → NN 融合 → X₀, Y₀, Z₀

 ═══════════════════════════════════════════════════
 STAGE 2: Intra-view temporal DIC (per camera, per frame)
 ═══════════════════════════════════════════════════
   cam0: ref→def₁, ref→def₂, ...  时序 DIC → u₀ᵗ, v₀ᵗ
   cam1: ref→def₁, ref→def₂, ...  时序 DIC → u₁ᵗ, v₁ᵗ
   ...

 ═══════════════════════════════════════════════════
 STAGE 3: Correspondence propagation + triangulation
 ═══════════════════════════════════════════════════
   (x, y) ──→ (x+u₀ᵗ, y+v₀ᵗ)  in cam0
   (xⱼ, yⱼ) ──→ (xⱼ+uⱼᵗ, yⱼ+vⱼᵗ)  in camⱼ
                    ↓
              三角化 → NN 融合 → Xt, Yt, Zt
              U,V,W = Xt - X₀
```

---

## Features

- **N-camera support** — Auto-discovers camera folders under `work_dir/images/`, no hardcoded camera count
- **COLMAP self-calibration** — No calibration targets needed; uses reference images for automatic SfM-based calibration
- **Three-stage efficient pipeline** — Cross-view DIC runs once (reference frame); temporal DIC within each camera is fast and parallelizable
- **NN-based multi-view fusion** — Coordinate MLP fits a smooth 3D surface from pairwise triangulations, with automatic outlier rejection and prefiltering
- **Full-field 3D output** — Displacement (U, V, W) and strain (exx, eyy, ezz, exy, exz, eyz) for every frame
- **PINN-based DIC engine** — Physics-informed neural networks for continuous displacement fields with FBPINN domain decomposition
- **Multiple network architectures** — AdaptiveFCN, AdaptiveSIREN, AdaptiveResNet, FourierNet
- **GPU / CPU toggle** — Simple flag in `DIC_importlib.py`
- **Multi-ROI support** — Automatic disconnected region detection from mask

---

## Installation

```bash
git clone https://github.com/lbd-hfut/PINN-Multi-DIC.git
cd PINN-Multi-DIC
pip install .
pip install pycolmap     # COLMAP Python bindings for self-calibration
```

Requires Python >= 3.10, JAX, and pycolmap.

---

## Quick Start

### 1. Prepare data

```
my_experiment/
├── images/
│   ├── cam1/
│   │   ├── 001.jpg      # Reference image (also calibration input)
│   │   ├── 002.jpg      # Deformed frame 1
│   │   ├── 003.jpg      # Deformed frame 2
│   │   └── ...
│   ├── cam2/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── camN/
│       └── ...
└── mask.png              # ROI mask (white=foreground, black=background)
```

- Each camera folder must have the **same number** of images
- The **first image** in each folder serves as both the DIC reference and the COLMAP calibration input
- Images must have sufficient texture and overlap between views for COLMAP SfM to succeed

### 2. Configure

```ini
# config/PINN-DIC-Mutil3D.txt

# work_dir:
C:/data/my_experiment

# mask_path:
C:/data/my_experiment/mask.png

# network:
AdaptiveFCN

# hidden_units:
[32, 32, 32, 32]

# loss_fun:
DIC_ZNSSD

# n_subdomains:
[2, 2]

# adam_epochs:
100

# seed_flag:
True
```

### 3. Run

```bash
python -m pinndicmulti.DIC_analysics3D
```

Or from Python:

```python
from pinndicmulti.DIC_analysics3D import main
main(
    dic_config_path="./config/PINN-DIC-Mutil3D.txt",
    seed_config_path="./config/Seed_Configuration.txt",
    fusion_config_path="./config/Fusion_Configuration.txt",
)
```

### 4. Outputs

```
my_experiment/
├── images/                        # Input (untouched)
├── calibration/
│   └── cameras.mat                # K, dist, R, t for all cameras
├── reconstruct/
│   ├── DIC_000.mat                # Reference 3D surface (X, Y, Z)
│   └── DIC_{001..N}.mat           # Deformed 3D surface per frame
├── deformation/
│   └── DEF_{001..N}.mat           # U, V, W, exx, eyy, ezz, exy, exz, eyz per frame
├── summaries/                     # TensorBoard logs
├── models/                        # Model checkpoints (.jax)
└── figs/                          # Visualization figures
    ├── 3D/                        # 3D surface plots (reference + per-frame deformed)
    ├── 2D/                        # 2D displacement & strain heatmaps
    ├── Disparity/                 # Per-pair DIC disparity field heatmaps
    └── seed/                      # Seed matching diagnostics
```

---

## Configuration

### DIC Solver Config (`config/PINN-DIC-Mutil3D.txt`)

| Key | Type | Default | Description |
|---|---|---|---|
| `work_dir` | `str` | — | Root working directory |
| `mask_path` | `str` | — | ROI mask image path |
| `network` | `str` | `AdaptiveFCN` | Network architecture |
| `hidden_units` | `list[int]` | `[32,32,32,32]` | Neurons per hidden layer |
| `loss_fun` | `str` | `DIC_ZNSSD` | Loss function (`DIC_MSE` or `DIC_ZNSSD`) |
| `n_subdomains` | `list[int]` | `[2, 2]` | FBPINN subdomains in (nx, ny) |
| `train_schedulers` | `str` | `AllActiveSchedulerND` | Subdomain activation scheduler |
| `spline_degree` | `int` | `5` | B-spline interpolation degree (1, 3, or 5) |
| `adam_epochs` | `int` | `100` | Adam training epochs |
| `adam_lr` | `float` | `0.01` | Adam learning rate |
| `dic_lr` | `float` | `0.01` | DIC-specific learning rate |
| `lbfgs_epochs` | `int` | `0` | L-BFGS refinement steps (0 = skip) |
| `seed_flag` | `bool` | `True` | Enable seed point initialization |
| `seed_train_epochs` | `int` | `0` | Seed pre-training epochs |
| `znssd_kernel_size` | `int` | `7` | ZNSSD kernel window size |
| `strain_window_len` | `int` | `5` | 3D strain smoothing window |
| `summary_freq` | `int` | `10` | Loss print frequency (epochs) |
| `test_freq` | `int` | `1000` | Test metric frequency (epochs) |
| `test_flag` | `bool` | `False` | Enable test metrics |
| `model_save_freq` | `int` | `10000` | Model save frequency (epochs) |
| `show_figures` | `bool` | `False` | Display result figures |
| `save_figures` | `bool` | `True` | Save result figures |

### Seed Config (`config/Seed_Configuration.txt`)

| Key | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `Sub_pixels` | Matching method (`Integer_pixels`, `Sub_pixels`, or `SIFT`) |
| `seeds_number` | `int` | `128` | Number of seed points |
| `max_workers` | `int` | `4` | Parallel threads |
| `coarse_subset_radius` | `int` | `28` | Coarse NCC subset radius (pixels) |
| `fine_subset_radius` | `int` | `9` | Fine IC-GN subset radius (pixels) |
| `max_iterations` | `int` | `50` | IC-GN max iterations |
| `cutoff_diffnorm` | `float` | `1e-5` | IC-GN convergence tolerance |
| `ncc_threshold` | `float` | `0.6` | NCC correlation threshold |
| `corrcoef_threshold` | `float` | `2.0` | ZNSSD correlation cutoff |
| `min_texture_std` | `int` | `5` | Minimum local texture std for seed |
| `lambda_reg` | `float` | `0.0` | IC-GN regularization |
| `plot_seed_flage` | `bool` | `True` | Enable seed match visualization |

### Fusion Config (`config/Fusion_Configuration.txt`)

Controls the coordinate-MLP that fuses multiple pairwise triangulation results into a smooth 3D surface. The network maps normalized ROI pixel coordinates `(x, y)` to 3D world coordinates `(X, Y, Z)`, trained on all pairwise triangulations with automatic outlier rejection.

**Network**

| Key | Type | Default | Description |
|---|---|---|---|
| `network` | `str` | `AdaptiveFCN` | Architecture from `DIC_networks.py` |
| `hidden_layers` | `int` | `3` | Number of hidden layers |
| `hidden_neurons` | `int` | `32` | Neurons per hidden layer |
| `output_mode` | `str` | `single` | `single` = one net for XYZ, `triple` = three independent nets |
| `fourier_mapping_size` | `int` | `64` | Fourier feature dim (FourierNet only) |
| `fourier_sigma_list` | `list[float]` | `[1,4,8,16]` | Multi-scale frequencies (FourierNet only) |

**Training**

| Key | Type | Default | Description |
|---|---|---|---|
| `adam_epochs` | `int` | `1000` | Adam training epochs |
| `adam_lr` | `float` | `0.001` | Adam learning rate |
| `lbfgs_epochs` | `int` | `0` | L-BFGS refinement steps (0 = skip) |
| `lbfgs_history_size` | `int` | `10` | L-BFGS history length |
| `lbfgs_maxls` | `int` | `15` | L-BFGS max line search iterations |
| `lbfgs_lr` | `float` | `1.0` | L-BFGS max step size |
| `lbfgs_tol` | `float` | `0.0` | L-BFGS convergence tolerance (0 = no early stop) |

**Data**

| Key | Type | Default | Description |
|---|---|---|---|
| `prefilter_outliers` | `bool` | `True` | Median-based outlier rejection before training |
| `outlier_threshold_sigma` | `float` | `3.0` | Rejection threshold (×σ from median) |
| `summary_freq` | `int` | `100` | Loss print frequency (epochs) |
| `save_figures` | `bool` | `True` | Save training loss curve |

---

## Project Structure

```
pinndicmulti/
├── DIC_analysics3D.py            # Main 3D pipeline (three-stage)
├── DIC_config.py                 # Config parsers + camera discovery
├── DIC_importlib.py              # Centralized imports + GPU/CPU toggle
├── segpinndic/                   # 2D PINN-DIC engine
│   ├── DIC_constants.py          # Domain/problem/network/decomposition wiring
│   ├── DIC_decompositions.py     # FBPINN rectangular/multilevel decomposition
│   ├── DIC_domains.py            # Rectangular domain + interior sampling
│   ├── DIC_networks.py           # FCN, SIREN, ResNet, FourierNet + adaptive variants
│   ├── DIC_plot_trainer.py       # 2D/3D visualization
│   ├── DIC_problem.py            # DIC_MSE, DIC_ZNSSD loss functions
│   ├── DIC_readImg.py            # Image I/O, B-spline buffers, MultiCamDataset
│   ├── DIC_schedulers.py         # Active subdomain schedulers
│   ├── DIC_seed_trainer.py       # Seed point supervised pre-training
│   ├── DIC_seedcalc.py           # Seed matching (K-means / SIFT + NCC + IC-GN)
│   ├── DIC_trainers.py           # PINNTrainer, FBPINNTrainer
│   ├── DIC_windows.py            # POU window functions (cosine, sigmoid, etc.)
│   └── utils/                    # I/O, JAX utils, logging, misc
└── reconstruction/               # 3D reconstruction pipeline
    ├── DIC_calibrate.py          # COLMAP multi-camera self-calibration
    ├── DIC_triangulation.py      # Pairwise + multi-camera triangulation
    ├── DIC_fusion_nn.py          # NN-based multi-view surface fusion (FusionTrainer)
    ├── DIC_fusion.py             # Point cloud fusion (average/median/robust)
    └── DIC_strain3Dcalc.py       # 3D strain via local least-squares
```

---

## Deformation Output Format

### `deformation/DEF_{frame:03d}.mat`

| Variable | Shape | Description |
|---|---|---|
| `X`, `Y`, `Z` | (H, W) | 3D coordinates of deformed surface |
| `U`, `V`, `W` | (H, W) | 3D displacement from reference |
| `exx`, `eyy`, `ezz` | (H, W) | Normal strain components |
| `exy`, `exz`, `eyz` | (H, W) | Shear strain components |

### `reconstruct/DIC_000.mat`

Reference surface `X, Y, Z` before deformation.

---

## Citation

```bibtex
@software{PINN-Multi-DIC,
  author = {Boda Li},
  title = {PINN-Multi-DIC: Multi-Camera PINN Framework for 3D Digital Image Correlation},
  year = {2025},
  url = {https://github.com/lbd-hfut/PINN-Multi-DIC}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with [JAX](https://github.com/google/jax), [Optax](https://github.com/google-deepmind/optax), [COLMAP](https://github.com/colmap/colmap), and inspired by [FBPINNs](https://github.com/mikkelbueholm/FBPINNs).
