"""Entry point for patient-held-out calibration and nested OOF protocol v2.

The historical Step 6 predictions remain unchanged. New runs require a separate
output directory and regenerate all learned components from their allowed roles.
"""

import os
from pathlib import Path
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clinical_training import main


if __name__ == "__main__":
    main()
