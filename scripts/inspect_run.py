from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(rows[: args.limit], indent=2))


if __name__ == "__main__":
    main()
