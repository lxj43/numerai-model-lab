"""Keep generated data and private experiment outputs in a configurable workspace."""
import os
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('NUMERAI_WORK_DIR', SOURCE)).expanduser().resolve()
