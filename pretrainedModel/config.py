"""
Configuration for memory optimization.

Set save_memory to True if you have limited GPU memory (< 12GB).
This will enable:
- Sliced attention mechanism (reduces memory usage during attention computation)
- Dynamic VRAM shifting (moves model between CPU/GPU during inference)

Set to False for faster inference on high-memory GPUs (>= 12GB).
"""

import os

# Memory optimization flag
# Can be overridden by environment variable: CONTROLNET_SAVE_MEMORY=1
save_memory = os.environ.get('CONTROLNET_SAVE_MEMORY', '0') == '1'

# Default: False (disabled for better performance)
# To enable: set environment variable before running:
#   export CONTROLNET_SAVE_MEMORY=1  (Linux/Mac)
#   set CONTROLNET_SAVE_MEMORY=1     (Windows)
