"""Launch Cadabrio."""

import sys
from pathlib import Path

# Add src to path so cadabrio package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cadabrio.__main__ import main

if __name__ == "__main__":
    main()
