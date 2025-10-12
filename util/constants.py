"""
Global constants for circle_art project.
These values are fixed across all experiments and should not be changed per-config.
"""

# ==================== Initialization Constants ====================
# Opacity initialization - fixed to ensure gradient flow through all layers
OPACITY_INIT_VALUE = -2.0  # sigmoid(-2.0) ≈ 0.12 (12% opacity)

# Color initialization noise
STD_C_INIT = 0.02  # Standard deviation for color initialization noise

# Variance-based sampling parameters
VARIANCE_WINDOW_SIZE = 7  # Window size for local variance computation
VARIANCE_BASE_PROB = 0.1  # Base probability for low-variance areas

# Unused legacy parameters (kept for reference, not used in code)
# V_INIT_BIAS = -5.0
# V_INIT_SLOPE = 3.0

# ==================== Optimization Constants ====================
LR_DEFAULT = 0.1
# Learning rate gains (relative to base learning rate)
LR_GAIN_X = 10.0
LR_GAIN_Y = 10.0
LR_GAIN_R = 10.0
LR_GAIN_V = 1.5
LR_GAIN_THETA = 1.0
LR_GAIN_C = 1.0

# Decay settings
DECAY_RATE = 0.99

# Alpha and blur settings
ALPHA_UPPER_BOUND = 1.0
BLUR_SIGMA = 1.0

# ==================== Preprocessing Constants ====================
# Background threshold for image processing
BG_THRESHOLD = 250  # Pixels >= this value are considered white background

# Preprocessing flags (usually disabled)
DO_EQUALIZE = False
DO_LOCAL_CONTRAST = False
DO_TONE_CURVE = False

# Vertical paddings
VERTICAL_PADDINGS = [0, 0]

# Local contrast enhancement defaults
LOCAL_CONTRAST_RADIUS = 2.0
LOCAL_CONTRAST_AMOUNT = 3.0

# Tone curve defaults
TONE_CURVE_IN_LOW = 50
TONE_CURVE_IN_HIGH = 200
TONE_CURVE_OUT_LOW = 30
TONE_CURVE_OUT_HIGH = 230

# ==================== Postprocessing Constants ====================
# PSD export scale factor
PSD_SCALE_FACTOR = 2.0

# Line width for vector export
LINEWIDTH = 3.0

# ==================== Other Constants ====================
# Random seed for reproducibility
DEFAULT_SEED = 42

# Primitive/SVG loading defaults
DEFAULT_OUTPUT_WIDTH = 128
CONVERT_TO_SVG_DEFAULT = True
REMOVE_PUNCTUATION_DEFAULT = False

# Renderer defaults
DEFAULT_RENDERER_TYPE = "tile"
DEFAULT_TILE_SIZE = 32

# Sequential processing defaults
SEQUENTIAL_ENABLED_DEFAULT = False
SEQUENTIAL_INPUT_TYPE_DEFAULT = "gif"

# Loss function defaults
DEFAULT_LOSS_CONFIG = {
    "type": "mse",  # Single loss type: "mse", "l1", "huber", "perceptual", "ssim", "edge", or "combined"
}

DEFAULT_LOSS_CONFIG_NO_BG = {
    "type": "combined",
    "components": [
        {"name": "mse", "weight": 1.0},
        {"name": "alpha", "weight": 2.0}
    ]
}


def apply_constants_to_config(config: dict) -> dict:
    """
    Apply constants as default values to config dictionary.
    This allows config files to override constants when needed.
    
    Args:
        config: Configuration dictionary loaded from JSON
        
    Returns:
        Modified config with constants applied as defaults
    """
    # Apply constants as defaults for initialization
    if "initialization" in config:
        config["initialization"].setdefault("std_c_init", STD_C_INIT)
        config["initialization"].setdefault("variance_window_size", VARIANCE_WINDOW_SIZE)
        config["initialization"].setdefault("variance_base_prob", VARIANCE_BASE_PROB)
    
    # Apply constants as defaults for preprocessing
    if "preprocessing" in config:
        config["preprocessing"].setdefault("bg_threshold", BG_THRESHOLD)
        config["preprocessing"].setdefault("do_equalize", DO_EQUALIZE)
        config["preprocessing"].setdefault("do_local_contrast", DO_LOCAL_CONTRAST)
        config["preprocessing"].setdefault("do_tone_curve", DO_TONE_CURVE)
        config["preprocessing"].setdefault("vertical_paddings", VERTICAL_PADDINGS)
        
        if "local_contrast" not in config["preprocessing"]:
            config["preprocessing"]["local_contrast"] = {}
        config["preprocessing"]["local_contrast"].setdefault("radius", LOCAL_CONTRAST_RADIUS)
        config["preprocessing"]["local_contrast"].setdefault("amount", LOCAL_CONTRAST_AMOUNT)
        
        if "tone_params" not in config["preprocessing"]:
            config["preprocessing"]["tone_params"] = {}
        config["preprocessing"]["tone_params"].setdefault("in_low", TONE_CURVE_IN_LOW)
        config["preprocessing"]["tone_params"].setdefault("in_high", TONE_CURVE_IN_HIGH)
        config["preprocessing"]["tone_params"].setdefault("out_low", TONE_CURVE_OUT_LOW)
        config["preprocessing"]["tone_params"].setdefault("out_high", TONE_CURVE_OUT_HIGH)

    # Apply constants as defaults for optimization
    if "optimization" in config:
        config["optimization"].setdefault("alpha_upper_bound", ALPHA_UPPER_BOUND)
        config["optimization"].setdefault("blur_sigma", BLUR_SIGMA)
        config["optimization"].setdefault("decay_rate", DECAY_RATE)
        config["optimization"].setdefault("renderer_type", DEFAULT_RENDERER_TYPE)
        config["optimization"].setdefault("tile_size", DEFAULT_TILE_SIZE)
        
        # Apply loss config based on whether background exists
        exist_bg = config.get("preprocessing", {}).get("exist_bg", True)
        if exist_bg:
            config["optimization"].setdefault("loss_config", DEFAULT_LOSS_CONFIG)
        else:
            config["optimization"].setdefault("loss_config", DEFAULT_LOSS_CONFIG_NO_BG)
        
        if "learning_rate" not in config["optimization"]:
            config["optimization"]["learning_rate"] = {}
        
        config["optimization"]["learning_rate"].setdefault("default", LR_DEFAULT)
        config["optimization"]["learning_rate"].setdefault("gain_x", LR_GAIN_X)
        config["optimization"]["learning_rate"].setdefault("gain_y", LR_GAIN_Y)
        config["optimization"]["learning_rate"].setdefault("gain_r", LR_GAIN_R)
        config["optimization"]["learning_rate"].setdefault("gain_v", LR_GAIN_V)
        config["optimization"]["learning_rate"].setdefault("gain_theta", LR_GAIN_THETA)
        config["optimization"]["learning_rate"].setdefault("gain_c", LR_GAIN_C)

    # Apply constants as defaults for postprocessing
    if "postprocessing" in config:
        config["postprocessing"].setdefault("psd_scale_factor", PSD_SCALE_FACTOR)
        config["postprocessing"].setdefault("linewidth", LINEWIDTH)
        config["postprocessing"].setdefault("output_folder", "./outputs/")
    
    # Apply constants as defaults for primitive section
    if "primitive" in config:
        config["primitive"].setdefault("output_width", DEFAULT_OUTPUT_WIDTH)
        config["primitive"].setdefault("bg_threshold", BG_THRESHOLD)
        config["primitive"].setdefault("convert_to_svg", CONVERT_TO_SVG_DEFAULT)
        config["primitive"].setdefault("remove_punctuation", REMOVE_PUNCTUATION_DEFAULT)
    
    # Apply constants as defaults for sequential section
    if "sequential" not in config:
        config["sequential"] = {}
    config["sequential"].setdefault("enabled", SEQUENTIAL_ENABLED_DEFAULT)
    config["sequential"].setdefault("input_type", SEQUENTIAL_INPUT_TYPE_DEFAULT)
    config["sequential"].setdefault("tile_size", DEFAULT_TILE_SIZE)
    
    # Apply seed default
    config.setdefault("seed", DEFAULT_SEED)
    
    return config
