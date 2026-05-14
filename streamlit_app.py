"""Repository-level launcher for the ICg-CaST Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from icg_cast.streamlit_app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
