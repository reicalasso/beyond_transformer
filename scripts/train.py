#!/usr/bin/env python3
"""Modern PULSE training launcher.

Delegates to :mod:`pulse.train`. All training logic lives there. To train
the legacy v3 prototype instead, run ``python -m pulse.legacy.train``.

Usage::

    python scripts/train.py
    python scripts/train.py --config configs/tinystories_small.yaml
    python scripts/train.py --hidden-size 768 --num-layers 12 --max-steps 50000
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pulse.train import main

if __name__ == "__main__":
    main()
