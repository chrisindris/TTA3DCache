from __future__ import annotations

import argparse
import json
from pathlib import Path

from tta3dcache.config import load_config
from tta3dcache.integration.cdviews_adapter import StaticCdViewsAdapter
from tta3dcache.integration.vlm_runner import MockVLMRunner
from tta3dcache.logging.jsonl_logger import JsonlLogger
from tta3dcache.logging.run_manifest import build_manifest, dump_manifest
from tta3dcache.pipeline.tta3dcache import run_mvp


def _parse_overrides(items: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use key=value.")
        key, raw_value = item.split("=", 1)
        if raw_value.lower() in {"true", "false"}:
            value: object = raw_value.lower() == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
        target = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})  # type: ignore[assignment]
        target[parts[-1]] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, _parse_overrides(args.overrides))

    example = {
        "scene_id": "demo-scene",
        "question_id": "demo-question",
        "question": "What is on the table?",
        "ranked_view_ids": ["v1", "v2", "v3", "v4", "v5"],
        "ranked_scores": [0.99, 0.98, 0.97, 0.4, 0.39],
        "base_view_ids": ["v1", "v2", "v3", "v4"],
        "metadata": {"nearby_view_ids": {"v3": ["v5"]}},
    }
    answer_map = {
        ("v1", "v2", "v3", "v4"): ("wrong answer", 0.4),
        ("v1", "v2", "v3", "v5"): ("wrong answer", 0.4),
        ("v1", "v2", "v4", "v5"): ("wrong answer", 0.4),
        ("v1", "v3", "v4", "v5"): ("correct answer", 0.95),
        ("v2", "v3", "v4", "v5"): ("correct answer", 0.95),
    }
    adapter = StaticCdViewsAdapter()
    runner = MockVLMRunner(answer_map, default_answer="wrong answer")
    result = run_mvp(example, cfg, cdviews_adapter=adapter, vlm_runner=runner, run_id="tta3dcache-demo")

    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(args.predictions or output_dir / "predictions.jsonl")
    logger = JsonlLogger(predictions_path)
    logger.write(
        {
            "run_id": result.run_id,
            "scene_id": result.scene_id,
            "question_id": result.question_id,
            "question": result.question,
            "base_answer": result.base_answer,
            "final_answer": result.final_answer,
            "adapted": result.adapted,
            "candidate_sets": [candidate.view_ids for candidate in result.candidate_sets],
            "candidate_answers": result.candidate_answers,
            "cache_scores": result.cache_scores,
            "prototype_scores": result.prototype_scores,
            "vlm_calls": result.vlm_calls,
            "final_correct": True,
        }
    )
    dump_manifest(output_dir / "run_manifest.json", build_manifest(result.run_id, cfg.to_dict(), ["python", "scripts/run_tta3dcache.py"], Path.cwd()))
    print(json.dumps({"predictions": str(predictions_path), "final_answer": result.final_answer}))


if __name__ == "__main__":
    main()
