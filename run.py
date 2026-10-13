import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "semantic segmentation"))

from combined_run import main


if __name__ == "__main__":
    main()
