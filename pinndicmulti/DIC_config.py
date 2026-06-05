from pinndicmulti.DIC_importlib import *

# ============================================
# 读取种子点配置文件
# ============================================
def seed_config_txt(path, verbose=True):
    """
    Parse a config.txt file like the provided monocular/stereo configuration.
    Each parameter is preceded by a comment line starting with '# key:'.
    Automatically infer types (int, float, list, bool, None, str).
    """

    config = {}
    current_key = None
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            # Comment or key line
            if line.startswith("#"):
                if ":" in line:
                    # Extract key name after '#'
                    parts = line[1:].split(":", 1)
                    current_key = parts[0].strip()
                    if verbose:
                        print(f"[line {lineno}] ⏳ Detected key '{current_key}' -> waiting for value...")
                else:
                    current_key = None
                    if verbose:
                        print(f"[line {lineno}] 📝 Ignored comment: {line}")
                continue
            # Value line
            if current_key is None:
                if verbose:
                    print(f"[line {lineno}] ⚠️ Value without key ignored: {line}")
                continue
            raw_value = line
            # Try to parse value safely
            if raw_value.lower() == 'null':
                value = None
            elif raw_value.lower() == 'true':
                value = True
            elif raw_value.lower() == 'false':
                value = False
            else:
                try:
                    value = ast.literal_eval(raw_value)
                except Exception:
                    value = raw_value  # leave as string if eval fails
            config[current_key] = value
            if verbose:
                print(f"[line {lineno}] ✅ Loaded key '{current_key}' = {value}")
            current_key = None  # reset after reading value
    if verbose:
        print("\n=== ✅ Configuration loaded successfully ===")
        for k, v in config.items():
            print(f"  {k}: {v}")
    return SimpleNamespace(**config)


def calibrate_config_txt(path, verbose=True):
    """
    Parse calibration config txt file.

    Supported keys:
    - calibrate1_dir (str / Path)
    - calibrate2_dir (str / Path)
    - pattern_type (str: 'chessboard' or 'circles')
    - length (float/int)
    - visualize (bool)
    """

    config = {}
    current_key = None

    with open(path, 'r', encoding='utf-8') as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            # --- 解析 key ---
            if line.startswith("#"):
                if ":" in line:
                    parts = line[1:].split(":", 1)
                    current_key = parts[0].strip()
                    if verbose:
                        print(f"[line {lineno}] ⏳ Detected key '{current_key}' -> waiting for value...")
                else:
                    current_key = None
                    if verbose:
                        print(f"[line {lineno}] 📝 Ignored comment: {line}")
                continue

            # --- 解析 value ---
            if current_key is None:
                if verbose:
                    print(f"[line {lineno}] ⚠️ Value without key ignored: {line}")
                continue

            raw_value = line

            # 类型推断
            if raw_value.lower() == 'null':
                value = None
            elif raw_value.lower() == 'true':
                value = True
            elif raw_value.lower() == 'false':
                value = False
            else:
                try:
                    value = ast.literal_eval(raw_value)
                except Exception:
                    value = raw_value  # 字符串

            config[current_key] = value

            if verbose:
                print(f"[line {lineno}] ✅ Loaded key '{current_key}' = {value}")

            current_key = None

    if verbose:
        print("\n=== ✅ Calibration configuration loaded successfully ===")
        for k, v in config.items():
            print(f"  {k}: {v}")

    return SimpleNamespace(**config)


ALL_KEYS_2D = [
    "input_dir", "output_dir", "n_subdomains",
    "hidden_units", "network", "spline_degree",
    "adam_epochs", "seed_flag", "seed_train_epochs",
    "adam_lr", "dic_lr", "summary_freq", "test_freq", "test_flag", "model_save_freq",
     "show_figures",  "save_figures",  "clear_output",
     "loss_fun", "train_schedulers", "znssd_kernel_size",
     "lbfgs_epochs", "lbfgs_history_size", "lbfgs_maxls", "lbfgs_lr",
     "seed_smooth_lambda", "seed_smooth_npoints", "seed_sd_threshold"
]
# ============================================
# 读取DIC配置文件
# ============================================
def DIC_2D_config_txt(path, required_keys=ALL_KEYS_2D, verbose=True):
    """
    Parse a config.txt file with lines like:
    # key: comment
    value
    Automatically infer types (int, float, list, None, bool, str).
    Checks all required_keys present.
    """
    config = {}
    current_key = None
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            raw_line = line
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if ":" in line:  # 有冒号的，识别为参数
                    parts = line[1:].split(":", 1)
                    current_key = parts[0].strip()
                    if verbose:
                        print(f"[line {lineno}] ⏳ Detected key '{current_key}' -> waiting for value...")
                else:
                    if verbose:
                        print(f"[line {lineno}] 📝 Comment ignored: {raw_line.strip()}")
                    current_key = None
            else:
                if current_key is None:
                    if verbose:
                        print(f"[line {lineno}] ⚠️ Value without key, ignored: {raw_line.strip()}")
                    continue
                raw_value = line
                if raw_value.lower() == 'null':
                    value = None
                elif raw_value.lower() == 'true':
                    value = True
                elif raw_value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = ast.literal_eval(raw_value)
                    except Exception:
                        value = raw_value
                config[current_key] = value
                if verbose:
                    print(f"[line {lineno}] ✅ Loaded key '{current_key}' = {value}")
                current_key = None

    # Check required keys
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    if verbose:
        print("\n=== ✅ All config loaded successfully ===")
        for k, v in config.items():
            print(f"  {k}: {v}")

    return SimpleNamespace(**config)


ALL_KEYS_3D = [
    "work_dir", "mask_path", "mask_camera",
    "adam_epochs", "strain_window_len", "seed_flag", "seed_train_epochs",
    "n_subdomains", "train_schedulers", "hidden_units", "network", "spline_degree", "loss_fun",
    "adam_lr", "summary_freq", "test_freq", "model_save_freq",
     "show_figures",  "save_figures",  "clear_output",
     "dic_lr", "test_flag", "znssd_kernel_size",
     "lbfgs_epochs", "lbfgs_history_size", "lbfgs_maxls", "lbfgs_lr",
     "seed_smooth_lambda", "seed_smooth_npoints", "seed_sd_threshold"
]

def get_work_subdir(work_dir, subdir):
    """Return path to a standard work subdirectory."""
    return os.path.join(work_dir, subdir)

def discover_cameras(work_dir):
    """Discover camera folders under work_dir/images/.

    Returns sorted list of folder names like ['cam1', 'cam2', ...].
    """
    images_dir = get_work_subdir(work_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    cam_dirs = sorted([
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ])
    if not cam_dirs:
        raise FileNotFoundError(f"No camera folders found under {images_dir}")
    return cam_dirs
# ============================================
# 读取DIC配置文件
# ============================================
def DIC_3D_config_txt(path, required_keys=ALL_KEYS_3D, verbose=True):
    """
    Parse a config.txt file with lines like:
    # key: comment
    value
    Automatically infer types (int, float, list, None, bool, str).
    Checks all required_keys present.
    """
    config = {}
    current_key = None
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            raw_line = line
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if ":" in line:  # 有冒号的，识别为参数
                    parts = line[1:].split(":", 1)
                    current_key = parts[0].strip()
                    if verbose:
                        print(f"[line {lineno}] ⏳ Detected key '{current_key}' -> waiting for value...")
                else:
                    if verbose:
                        print(f"[line {lineno}] 📝 Comment ignored: {raw_line.strip()}")
                    current_key = None
            else:
                if current_key is None:
                    if verbose:
                        print(f"[line {lineno}] ⚠️ Value without key, ignored: {raw_line.strip()}")
                    continue
                raw_value = line
                if raw_value.lower() == 'null':
                    value = None
                elif raw_value.lower() == 'true':
                    value = True
                elif raw_value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = ast.literal_eval(raw_value)
                    except Exception:
                        value = raw_value
                config[current_key] = value
                if verbose:
                    print(f"[line {lineno}] ✅ Loaded key '{current_key}' = {value}")
                current_key = None

    # Check required keys
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    if verbose:
        print("\n=== ✅ All config loaded successfully ===")
        for k, v in config.items():
            print(f"  {k}: {v}")

    return SimpleNamespace(**config)


ALL_KEYS_FUSION = [
    "network", "hidden_layers", "hidden_neurons", "output_mode",
    "fourier_mapping_size", "fourier_sigma_list",
    "adam_epochs", "adam_lr",
    "lbfgs_epochs", "lbfgs_history_size", "lbfgs_maxls", "lbfgs_lr", "lbfgs_tol",
    "prefilter_outliers", "outlier_threshold_sigma",
    "summary_freq", "save_figures",
]

# ============================================
# 读取多视角融合配置文件
# ============================================
def fusion_config_txt(path, required_keys=ALL_KEYS_FUSION, verbose=True):
    """
    Parse fusion config txt file.

    Supported sections:
    - Network: network, hidden_layers, hidden_neurons, output_mode, fourier_*
    - Training: adam_epochs, adam_lr, lbfgs_*
    - Data: prefilter_outliers, outlier_threshold_sigma
    - Output: summary_freq, save_figures
    """
    config = {}
    current_key = None
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if ":" in line:
                    parts = line[1:].split(":", 1)
                    current_key = parts[0].strip()
                    if verbose:
                        print(f"[line {lineno}] ⏳ Detected key '{current_key}' -> waiting for value...")
                else:
                    current_key = None
                    if verbose:
                        print(f"[line {lineno}] 📝 Comment ignored: {raw_line.strip()}")
                continue
            if current_key is None:
                if verbose:
                    print(f"[line {lineno}] ⚠️ Value without key ignored: {line}")
                continue
            raw_value = line
            if raw_value.lower() == 'null':
                value = None
            elif raw_value.lower() == 'true':
                value = True
            elif raw_value.lower() == 'false':
                value = False
            else:
                try:
                    value = ast.literal_eval(raw_value)
                except Exception:
                    value = raw_value
            config[current_key] = value
            if verbose:
                print(f"[line {lineno}] ✅ Loaded key '{current_key}' = {value}")
            current_key = None

    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing required fusion config keys: {missing}")

    if verbose:
        print("\n=== ✅ Fusion configuration loaded successfully ===")
        for k, v in config.items():
            print(f"  {k}: {v}")

    return SimpleNamespace(**config)


if __name__ == "__main__":
    # 测试读取配置文件
    # config = DIC_2D_config_txt("./config/PINN-DIC-2D.txt")
    # for i in range(5):
    #     print()

    # config = DIC_3D_config_txt("./config/PINN-DIC-3D.txt")

    config = calibrate_config_txt("./config/Calibration_Configuration.txt")