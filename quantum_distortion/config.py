DEFAULT_SAMPLE_RATE = 48000

DEFAULT_KEY = "D"
DEFAULT_SCALE = "minor"

DEFAULT_SNAP_STRENGTH = 1.0
DEFAULT_SMEAR = 0.1
DEFAULT_BIN_SMOOTHING = True

DEFAULT_DISTORTION_MODE = "wavefold"

DEFAULT_LIMITER_ON = True
DEFAULT_LIMITER_CEILING_DB = -1.0
DEFAULT_DRY_WET = 1.0

# Preview render mode: limits processing to first N seconds for faster iteration
# Set DSP_PREVIEW_MODE=1 environment variable to enable, or pass preview_enabled=True
PREVIEW_ENABLED_DEFAULT = False  # Default to full render for production
PREVIEW_MAX_SECONDS = 10.0  # Process only first 10 seconds in preview mode
