
from __future__ import annotations
import argparse
import os
import sys

# Allow running as a script from repo root or elsewhere
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import load_data, build_and_solve_lp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data"))
    args = ap.parse_args()
    data = load_data(args.data_dir)
    build_and_solve_lp(data, verbose=True)


if __name__ == "__main__":
    main()
