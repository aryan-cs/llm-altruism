from __future__ import annotations

import os


# Force a production-safe headless backend before test modules import pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")
