from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _validate_range(name: str, value: float, lower: float, upper: float) -> None:
    if not (lower <= value <= upper):
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {value}")


def _coerce_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(slots=True)
class DatasetConfig:
    name: str = "mock"
    split: str = "test"
    limit: int | None = None


@dataclass(slots=True)
class CdViewsConfig:
    views_per_set: int = 4
    preserve_original_prompt: bool = True


@dataclass(slots=True)
class CandidateSetConfig:
    num_sets: int = 5
    diversity_lambda: float = 0.5
    include_random_control: bool = False
    max_ranked_pool: int = 16


@dataclass(slots=True)
class AnswerConfig:
    equivalence_mode: str = "exact_normalized"
    embedding_threshold: float = 0.85
    canonicalize_spatial_terms: bool = True


@dataclass(slots=True)
class FeatureConfig:
    enabled: bool = False
    pooling: str = "weighted_mean"
    normalize: bool = True


@dataclass(slots=True)
class ConfidenceConfig:
    source_priority: tuple[str, ...] = ("model_probability", "agreement", "self_report")
    minimum_cache_confidence: float = 0.35


@dataclass(slots=True)
class CacheConfig:
    enabled: bool = True
    max_size: int = 32
    pruning: str = "confidence_diversity"
    temperature: float = 1.0
    similarity_temperature: float = 1.0
    similarity_threshold: float = 0.75
    use_negative_cache: bool = False


@dataclass(slots=True)
class PrototypeConfig:
    enabled: bool = False
    momentum: float = 0.9
    minimum_support: int = 2


@dataclass(slots=True)
class WeightConfig:
    confidence_exponent: float = 1.0
    diversity_exponent: float = 1.0
    criticality_exponent: float = 1.0
    overlap_exponent: float = 1.0
    instability_exponent: float = 1.0


@dataclass(slots=True)
class FusionConfig:
    lambda_base: float = 1.0
    lambda_cache: float = 1.0
    lambda_proto: float = 1.0
    lambda_instability: float = 0.5
    fallback_entropy_threshold: float = 0.8


@dataclass(slots=True)
class GatingConfig:
    enabled: bool = True
    agreement_threshold: float = 0.67
    entropy_threshold: float = 0.75
    initial_num_sets: int = 3


@dataclass(slots=True)
class LoggingConfig:
    output_dir: str = "outputs/full_tta3dcache"
    save_features: bool = False


@dataclass(slots=True)
class TTA3DCacheConfig:
    seed: int = 42
    device: str = "cpu"
    model_id: str = "mock-vlm"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cdviews: CdViewsConfig = field(default_factory=CdViewsConfig)
    candidate_sets: CandidateSetConfig = field(default_factory=CandidateSetConfig)
    answers: AnswerConfig = field(default_factory=AnswerConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.cdviews.views_per_set <= 0:
            raise ValueError("cdviews.views_per_set must be positive")
        if self.candidate_sets.num_sets <= 0:
            raise ValueError("candidate_sets.num_sets must be positive")
        if self.candidate_sets.max_ranked_pool <= 0:
            raise ValueError("candidate_sets.max_ranked_pool must be positive")
        _validate_range("confidence.minimum_cache_confidence", self.confidence.minimum_cache_confidence, 0.0, 1.0)
        _validate_range("cache.temperature", self.cache.temperature, 0.0, 100.0)
        _validate_range("cache.similarity_threshold", self.cache.similarity_threshold, 0.0, 1.0)
        _validate_range("cache.similarity_temperature", self.cache.similarity_temperature, 0.0, 100.0)
        _validate_range("prototype.momentum", self.prototype.momentum, 0.0, 1.0)
        _validate_range("fusion.fallback_entropy_threshold", self.fusion.fallback_entropy_threshold, 0.0, 1.0)
        _validate_range("gating.agreement_threshold", self.gating.agreement_threshold, 0.0, 1.0)
        _validate_range("gating.entropy_threshold", self.gating.entropy_threshold, 0.0, 1.0)
        if self.gating.initial_num_sets <= 0:
            raise ValueError("gating.initial_num_sets must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, payload: Mapping[str, Any]) -> Any:
    values = {}
    for field_info in instance.__dataclass_fields__.values():
        current_value = getattr(instance, field_info.name)
        if field_info.name not in payload:
            values[field_info.name] = current_value
            continue
        raw_value = payload[field_info.name]
        if hasattr(current_value, "__dataclass_fields__") and isinstance(raw_value, Mapping):
            values[field_info.name] = _merge_dataclass(current_value, raw_value)
        else:
            values[field_info.name] = raw_value
    return type(instance)(**values)


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> TTA3DCacheConfig:
    config = TTA3DCacheConfig()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        config = _merge_dataclass(config, _coerce_mapping(payload))
    if overrides:
        config = _merge_dataclass(config, _coerce_mapping(overrides))
    config.validate()
    return config
