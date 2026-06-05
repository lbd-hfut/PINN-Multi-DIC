# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
pip install -e .                          # Install in development mode
python -m pinndicmulti.DIC_analysics3D    # Run 3D pipeline with default config paths
```

No linting or test framework is configured. The project uses `hatchling` (`pyproject.toml`).

## GPU / CPU Toggle

Set `use_gpu = False` in `pinndicmulti/DIC_importlib.py` to switch to CPU via `JAX_PLATFORM_NAME`. All third-party imports are centralized in this file — every other module imports from here.

## Architecture

This is a **multi-view stereo DIC** framework. It extends a 2D PINN-based DIC engine (`segpinndic/`) with a 3D reconstruction pipeline (`reconstruction/`). The 2D core is a derivative of the standalone [SegPINN-DIC](https://github.com/lbd-hfut/SegPINN-DIC) project.

### Package Layout

```
pinndicmulti/
├── DIC_analysics3D.py        # Main entry point — 3D pipeline orchestrator
├── DIC_config.py             # Config parsers (Seed, DIC-2D, DIC-3D, Calibration)
├── DIC_importlib.py          # Centralized imports + GPU/CPU toggle + random seed
├── segpinndic/               # 2D PINN-DIC core (adapted from SegPINN-DIC)
│   ├── DIC_constants.py      # Wires domain/problem/network/decomposition/scheduler per ROI
│   ├── DIC_decompositions.py # FBPINN domain decomposition (rectangular, multilevel)
│   ├── DIC_domains.py        # Rectangular domain + interior point sampling
│   ├── DIC_networks.py       # NN architectures (FCN, SIREN, ResNet, FourierNet, adaptive variants)
│   ├── DIC_plot_trainer.py   # 2D/3D visualization (uv, uvw, xyz plots)
│   ├── DIC_problem.py        # Loss functions (DIC_MSE, DIC_ZNSSD)
│   ├── DIC_readImg.py        # Image I/O, B-spline buffers, ImgDataset2D, ImgDataset3D
│   ├── DIC_schedulers.py     # Active subdomain schedulers
│   ├── DIC_seedcalc.py       # Seed matching (K-means → NCC → IC-GN)
│   ├── DIC_trainers.py       # PINNTrainer and FBPINNTrainer
│   ├── DIC_windows.py        # Window functions for POU (cosine, sigmoid, etc.)
│   └── utils/                # io, jax_util (partition/combine), logger, other
└── reconstruction/           # 3D stereo pipeline
    ├── DIC_calibrate.py      # Stereo calibration (chessboard / circles)
    ├── DIC_triangulation.py  # Triangulation from disparity + calibration
    └── DIC_strain3Dcalc.py   # 3D strain via local least-squares
```

### Data Flow (3D Pipeline)

```
Config files (.txt) → DIC_analysics3D.main()
  → ImgDataset3D (load cam1 + cam2 sequences, interleave)
  → stereo_calibrate() (compute K1,K2,R,T,P1,P2, save .mat)
  → For each interleaved image pair (even=disparity, odd=temporal):
      → CalcSeeds (NCC + IC-GN on cam1 ref → current def)
      → PINNTrainer / FBPINNTrainer (2D DIC optimization)
      → If even (disparity frame):
          → triangulation(coords, Utemporal, Udisparity) → 3D points
          → Frame 0: save Xworld0 (initial topography)
          → Frame N>0: compute U,V,W = Xworld - Xworld0, then 3D strain
      → Save .mat files (disparity, temporal, 3D) + figures
```

### Image Sequence Convention (3D)

`ImgDataset3D` builds an interleaved deformation sequence:
```
Index 0:  cam2_ref       (disparity — stereo match to cam1_ref)
Index 1:  cam1_def_1     (temporal — same camera, different time)
Index 2:  cam2_def_1     (disparity)
Index 3:  cam1_def_2     (temporal)
...
```
Even indices (0,2,4,…) produce disparity fields used for triangulation. Odd indices (1,3,5,…) produce temporal fields used to track motion over time. Triangulation reconstructs 3D coordinates from: `(Xref + Utemporal, Xref + Udisparity)` on left/right cameras.

### Core Abstractions (shared with SegPINN-DIC)

All core classes follow a **functional-JAX pattern**: `@staticmethod` methods with explicit `params` dicts structured as `{"static": {...}, "trainable": {...}}`.

| Component | File | Role |
|---|---|---|
| `Domain` | `segpinndic/DIC_domains.py` | Sample interior points, normalize coordinates |
| `Problem` | `segpinndic/DIC_problem.py` | Loss functions (`DIC_MSE`, `DIC_ZNSSD`) |
| `Network` | `segpinndic/DIC_networks.py` | FCN, AdaptiveFCN, SIREN, AdaptiveSIREN, ResNet, AdaptiveResNet, FourierNet |
| `Decomposition` | `segpinndic/DIC_decompositions.py` | FBPINN subdomain routing + POU (norm/unnorm/window per subdomain) |
| `ActiveScheduler` | `segpinndic/DIC_schedulers.py` | Controls active/fixed subdomains per training step |

### Trainer Design

- **`PINNTrainer`**: Single network over entire ROI. Adam + optional L-BFGS refinement. AOT-compiles the update step.
- **`FBPINNTrainer`**: Multiple subdomain networks with partition-of-unity combining. Re-AOT-compiles when the active subdomain set changes per the scheduler.

Both accept a `dim` parameter in `train(dim=3)` that controls output directory structure (2D vs 3D-specific subdirectories).

### B-Spline Interpolation

FFT-based coefficient precomputation. The key buffer is `QKBQKT_def_DIC`: a `(H, W, degree+1, degree+1)` tensor enabling O(1) per-pixel spline evaluation at warped coordinates. Default degree is 5.

### 3D Triangulation

Calibration is loaded from a MATLAB `.mat` file (flat format or `stereoParameters` object). Supports both pre-computed and on-the-fly calibration via `stereo_calibrate()`. Triangulation undistorts points with `cv2.undistortPoints`, then uses `cv2.triangulatePoints`.

### 3D Strain Computation

Local least-squares fit over a `strain_window_len` window using normal equations (`AtA @ x = Atb`). Produces 6 strain components: `exx, eyy, ezz, exy, exz, eyz`.

## Configuration

Three text config files with `# key: comment` / `value` format:

| Config | Parser | Purpose |
|---|---|---|
| `config/PINN-DIC-Mutil3D.txt` | `DIC_3D_config_txt` | cam dirs, network, training, strain params |
| `config/Seed_Configuration.txt` | `seed_config_txt` | Seed matching method, counts, subset radii |
| `config/Calibration_Configuration.txt` | `calibrate_config_txt` | Calibration image dirs, pattern type/size |

## Key Notes

- **`BufferManager` is mutable global state**: Holds images, masks, B-spline coefficient tensors. Cleared/rebuilt per image pair. All modules reference it directly.
- **JIT static/dynamic separation**: The `partition()`/`combine()` pattern in `segpinndic/utils/jax_util.py` is essential — static params (shapes, image data) are separated from dynamic params (trainable weights) before JIT.
- **Multi-ROI**: Disconnected mask regions are auto-labeled via `scipy.ndimage.label` and processed independently. Results are assembled back into full-image arrays.
- **The 2D core in `segpinndic/`** is a bundled subset of SegPINN-DIC, adapted for `pinndicmulti` namespace imports. Changes from standalone SegPINN-DIC include: `ImgDataset3D` (dual-camera loading), `get_outdirs(dim)`/`clear_outdirs(dim)` for 3D directory layout, and `result_uvw_plot`/`result_xyz_plot` for 3D visualization.
