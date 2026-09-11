"""Download Jeff Sackmann ATP/WTA match-stat CSVs used for serve/return features."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sackmann_stats import download_match_stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tour", choices=["atp", "wta"], nargs="?", default="atp")
    parser.add_argument("--from-year", type=int, default=2000)
    parser.add_argument("--to-year", type=int, default=datetime.now().year)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = download_match_stats(args.from_year, args.to_year, force=args.force, tour=args.tour)
    print(f"Sackmann {args.tour.upper()}: {len(paths)} fichier(s) disponibles dans data/sackmann")
    for path in paths[-5:]:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
