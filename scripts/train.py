#!/usr/bin/env python3
"""
PULSE v3 training launcher.

Delegates entirely to src/pulse/train.py — all training logic lives there.

Usage:
    python scripts/train.py                                      # defaults
    python scripts/train.py --config configs/rtx4090_tinystories.yaml
    python scripts/train.py --hidden-size 512 --num-layers 6    # CLI overrides
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pulse.train import main

if __name__ == "__main__":
    main()
