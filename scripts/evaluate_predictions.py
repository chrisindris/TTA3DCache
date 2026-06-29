from __future__ import annotations

import argparse
import json
from pathlib import Path

from tta3dcache.evaluation.bootstrap import paired_bootstrap_delta


def _load_predictions(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()
    predictions = _load_predictions(Path(args.predictions))
    accuracy = sum(1 for row in predictions if row.get("final_correct")) / len(predictions) if predictions else 0.0
    report: dict[str, object] = {"accuracy": accuracy, "count": len(predictions)}
    if args.compare:
        compare = _load_predictions(Path(args.compare))
        base = [1 if row.get("final_correct") else 0 for row in compare]
        adapted = [1 if row.get("final_correct") else 0 for row in predictions]
        delta, ci = paired_bootstrap_delta(base, adapted)
        report.update({"delta": delta, "bootstrap_ci": ci})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
