# Implementation Prompt: TTA3DCache

You are an expert research software engineer working on a computer vision / multimodal learning codebase. Your task is to implement a training-free test-time adaptation system for multi-view 3D question answering.

The template repository is named `TTA3DCache` and contains three Git submodules:

```text
src/tta3dcache/cdViews
src/tta3dcache/TDA
src/tta3dcache/Uni-Adapter
```

These are forks of:

- `cdViews`: the host 3D question-answering pipeline and source of question-conditioned view selection and view non-maximum suppression.
- `TDA`: the main reference for a training-free dynamic key-value cache.
- `Uni-Adapter`: the main reference for dynamic prototypes, confidence/entropy-aware fusion, and training-free online adaptation in 3D vision-language models.

Do not attempt to combine the three repositories wholesale. Use `cdViews` as the host pipeline and port only the minimal reusable ideas from `TDA` and `Uni-Adapter` into clean, original modules under the top-level `tta3dcache` package.

The target system is called **TTA3DCache**.

---

## 1. Research objective

Implement a plug-in test-time adaptation layer for multi-view 3D question answering.

Given:

- a 3D scene represented by multiple RGB views;
- a natural-language question;
- the ranked and diversity-filtered views returned by `cdViews`;
- a frozen vision-language model used by `cdViews`;

the method must:

1. construct multiple plausible candidate view sets;
2. query the frozen VLM on each candidate view set;
3. build a temporary cache of question-conditioned view-set evidence;
4. group semantically equivalent candidate answers;
5. weight each answer according to confidence, view diversity, view overlap, and consistency;
6. optionally maintain answer-conditioned feature prototypes;
7. fuse the base `cdViews` answer with cache/prototype evidence;
8. return the adapted answer without updating the VLM parameters.

The first implementation must use a **question-local cache** that is reset for every scene-question pair.

A later optional mode may keep a cache across questions from the same scene, but scene-level persistence must not be enabled by default.

---

## 2. Core scientific hypothesis

The same scene views can be informative for one question and irrelevant for another. Therefore, the cache key must represent the joint relationship between the question and the selected view set rather than the visual views alone.

Each cache entry should conceptually contain:

```python
key = question-conditioned view-set feature
value = normalized candidate answer
metadata = {
    "diversity": ...,
    "uncertainty": ...,
    "view_overlap": ...,
    "prompt_type": ...,
    "cdviews_score": ...,
}
```

The roles are:

- **Key:** identifies the question-view evidence that produced a prediction.
- **Value:** identifies the semantic answer supported by that evidence.
- **Metadata:** determines whether this evidence is reliable and independent enough to receive substantial weight.

The method must be demonstrably stronger than naive majority voting. The implementation must therefore preserve all information needed to compare:

- base `cdViews`;
- random or extra-view augmentation;
- majority vote;
- confidence-weighted vote;
- diversity-weighted vote;
- TDA-style cache;
- prototype cache;
- full TTA3DCache.

---

## 3. Constraints

### 3.1 Training-free first

The primary implementation must not:

- fine-tune the VLM;
- update LoRA parameters;
- backpropagate through the VLM;
- train a new neural network;
- require labels at test time.

Frozen auxiliary encoders are allowed.

### 3.2 Minimal intrusion

Do not scatter invasive changes throughout the `cdViews` submodule.

Preferred strategy:

- create adapters/wrappers around `cdViews`;
- preserve upstream behavior;
- add small integration hooks only where necessary;
- document every submodule modification.

### 3.3 Reproducibility

Every run must record:

- random seed;
- configuration;
- Git commit hashes of all three submodules;
- model identifier;
- dataset and split;
- question and scene IDs;
- selected views;
- all candidate view sets;
- all raw and normalized answers;
- all component scores;
- final prediction;
- number of VLM calls;
- runtime.

### 3.4 Modular ablations

Every major component must be independently switchable through configuration.

---

## 4. Required repository structure

Create or complete the following structure outside the three submodules:

```text
TTA3DCache/
├── pyproject.toml
├── README.md
├── configs/
│   ├── baseline.yaml
│   ├── majority_vote.yaml
│   ├── answer_cache.yaml
│   ├── prototype_cache.yaml
│   └── full_tta3dcache.yaml
├── scripts/
│   ├── run_cdviews_baseline.py
│   ├── run_tta3dcache.py
│   ├── evaluate_predictions.py
│   └── inspect_run.py
├── src/
│   └── tta3dcache/
│       ├── __init__.py
│       ├── cdViews/
│       ├── TDA/
│       ├── Uni-Adapter/
│       ├── integration/
│       │   ├── __init__.py
│       │   ├── cdviews_adapter.py
│       │   ├── vlm_runner.py
│       │   └── dataset_adapter.py
│       ├── view_sampling/
│       │   ├── __init__.py
│       │   ├── candidate_sets.py
│       │   ├── perturbations.py
│       │   └── diversity.py
│       ├── cache/
│       │   ├── __init__.py
│       │   ├── entries.py
│       │   ├── answer_cache.py
│       │   ├── prototype_cache.py
│       │   └── pruning.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── pooled_view_encoder.py
│       │   └── question_conditioning.py
│       ├── answers/
│       │   ├── __init__.py
│       │   ├── normalization.py
│       │   ├── equivalence.py
│       │   └── confidence.py
│       ├── scoring/
│       │   ├── __init__.py
│       │   ├── overlap.py
│       │   ├── consistency.py
│       │   ├── fusion.py
│       │   └── uncertainty.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── baseline.py
│       │   └── tta3dcache.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── qa_metrics.py
│       │   ├── efficiency.py
│       │   └── bootstrap.py
│       ├── logging/
│       │   ├── __init__.py
│       │   ├── jsonl_logger.py
│       │   └── run_manifest.py
│       └── config.py
└── tests/
    ├── test_answer_normalization.py
    ├── test_candidate_sets.py
    ├── test_answer_cache.py
    ├── test_prototype_cache.py
    ├── test_overlap_weighting.py
    ├── test_fusion.py
    └── test_pipeline_smoke.py
```

You may adjust file names if the existing template already imposes a compatible structure, but keep the same separation of responsibilities.

---

## 5. Inspect the submodules before coding

Before implementation:

1. inspect the `cdViews` inference entry point;
2. locate:
   - dataset loading;
   - scene/question representation;
   - view ranking;
   - `viewSelector`;
   - `viewNMS`;
   - VLM prompting;
   - answer parsing;
   - evaluation;
3. inspect TDA’s:
   - cache entry representation;
   - positive and negative caches, if present;
   - confidence filtering;
   - fixed-size queue logic;
   - cache-logit fusion;
4. inspect Uni-Adapter’s:
   - prototype representation;
   - prototype updates;
   - entropy/confidence weighting;
   - cache or prototype fusion;
   - optional graph smoothing.

Write a concise architectural note in:

```text
docs/submodule_mapping.md
```

The note must identify the exact files/classes/functions used as references and explain what will and will not be ported.

Do not copy large blocks blindly. Reimplement the required abstractions cleanly under `src/tta3dcache`.

---

## 6. Required data structures

Use typed dataclasses or equivalent validated structures.

### 6.1 Candidate view set

```python
@dataclass(frozen=True)
class CandidateViewSet:
    candidate_id: str
    view_ids: tuple[str, ...]
    generation_type: str
    cdviews_scores: tuple[float, ...]
    diversity_score: float
    overlap_with_base: float
    metadata: dict[str, Any]
```

`generation_type` should support at least:

- `cdviews_default`;
- `top_critical`;
- `diversity_heavy`;
- `leave_one_out`;
- `nearby_frame_jitter`;
- `random_control`.

### 6.2 Candidate answer

```python
@dataclass
class CandidateAnswer:
    candidate_id: str
    raw_answer: str
    normalized_answer: str
    semantic_cluster_id: str
    raw_response: str | None
    rationale: str | None
    confidence: float
    uncertainty: float
    view_ids: tuple[str, ...]
    diversity_score: float
    overlap_score: float
    cdviews_score: float
    prompt_type: str
    feature: Tensor | None
    metadata: dict[str, Any]
```

### 6.3 Cache entry

```python
@dataclass
class CacheEntry:
    key: Tensor | None
    value: str
    confidence: float
    uncertainty: float
    diversity: float
    overlap: float
    cdviews_score: float
    prompt_type: str
    candidate_id: str
    metadata: dict[str, Any]
```

---

## 7. cdViews integration

Implement a `CdViewsAdapter` that exposes a stable interface independent of internal cdViews details.

Suggested interface:

```python
class CdViewsAdapter(Protocol):
    def prepare_example(self, example: Any) -> PreparedExample:
        ...

    def select_views(
        self,
        scene: Any,
        question: str,
    ) -> CdViewsSelection:
        ...

    def answer(
        self,
        question: str,
        view_ids: Sequence[str],
        prompt_type: str = "short_answer",
    ) -> VLMResponse:
        ...
```

`CdViewsSelection` must expose:

```python
@dataclass
class CdViewsSelection:
    ranked_view_ids: list[str]
    ranked_scores: list[float]
    nms_view_ids: list[str]
    nms_scores: list[float]
    overlap_matrix: np.ndarray | None
    raw_metadata: dict[str, Any]
```

If cdViews does not expose all of this directly, add the smallest possible hook and document it.

The unmodified baseline path must still be runnable.

---

## 8. Candidate view-set generation

Implement deterministic candidate generation given a seed.

Required variants:

### 8.1 Default

Use the original `cdViews` NMS-filtered top-k set.

### 8.2 Top critical

Use the highest-ranked views before diversity filtering.

### 8.3 Diversity-heavy

Select views greedily to maximize:

```text
criticality + lambda_diversity * dissimilarity
```

Prefer reuse of cdViews overlap or NMS information.

### 8.4 Leave-one-out

Create variants by removing one view from the base set and replacing it with the next ranked non-selected view where possible.

### 8.5 Nearby-frame jitter

Replace selected frames with temporally or spatially nearby alternatives where dataset metadata permits.

### 8.6 Random control

Sample a random valid view set of equal size. This is an experimental control, not part of the full method.

Deduplicate candidate sets by sorted view IDs.

Configuration must control:

```yaml
candidate_sets:
  num_sets: 5
  views_per_set: 4
  include_default: true
  include_top_critical: true
  include_diversity_heavy: true
  include_leave_one_out: true
  include_jitter: true
  include_random_control: false
```

---

## 9. Frozen VLM inference

Wrap the cdViews VLM invocation behind `VLMRunner`.

Support at least two prompt types:

### 9.1 Short answer

```text
You are answering a question about a 3D scene using the provided views.
Return only a short answer phrase.

Question: {question}
Answer:
```

### 9.2 Structured evidence

Request parseable JSON containing:

```json
{
  "answer": "short answer phrase",
  "evidence": "one concise sentence",
  "confidence": "low|medium|high"
}
```

The main benchmark path should use a prompt as close as possible to the original cdViews prompt.

The structured prompt is primarily for analysis, confidence estimation, and ablation.

Implement robust parsing with fallback to raw text.

---

## 10. Answer normalization and semantic equivalence

Open-ended answers must not be grouped only by exact string equality.

Implement a layered normalizer:

1. Unicode normalization;
2. lowercase;
3. whitespace and punctuation cleanup;
4. article removal;
5. number normalization;
6. common contraction normalization;
7. optional lemmatization;
8. canonicalization of spatial terms.

Examples that should be equivalent where context allows:

```text
"on the left side of the table"
"left of the table"
"to the table's left"
```

Do not over-normalize distinct answers such as:

```text
"left"
"right"
```

Implement two equivalence modes:

- `exact_normalized`;
- `embedding_cluster`.

For `embedding_cluster`, use a frozen text encoder and cosine similarity with a configurable threshold. Cache the text embeddings.

Do not use an external LLM judge in the core method. An LLM judge may be added only as an optional offline analysis tool.

---

## 11. Question-conditioned view-set features

Implement a pluggable `QuestionConditionedFeatureEncoder`.

Start with a simple, transparent implementation.

### 11.1 Visual pooling

For views \(v_1,\ldots,v_k\):

```python
visual_feature = weighted_mean(
    image_encoder(v_i),
    weights=normalized_cdviews_scores,
)
```

Support:

- mean pooling;
- score-weighted mean pooling;
- max pooling as an ablation.

### 11.2 Question conditioning

Support at least:

#### Concatenation

```python
key = normalize(
    concatenate(
        text_projection(question_embedding),
        visual_projection(visual_feature),
    )
)
```

#### Gated fusion

```python
gate = sigmoid(Wq(question_embedding))
key = normalize(
    gate * visual_projection(visual_feature)
    + (1 - gate) * text_projection(question_embedding)
)
```

If avoiding learned parameters, use fixed dimensionality reduction or equal-dimensional encoders and a deterministic weighted sum.

Because the primary method is training-free, do not introduce randomly initialized trainable projections into the default configuration.

Preferred default:

```python
key = normalize(
    alpha * normalize(question_embedding)
    + (1 - alpha) * normalize(visual_feature)
)
```

only when the text and visual encoder share an embedding space, such as CLIP or SigLIP.

Otherwise use concatenation followed by normalization.

Make the encoder configurable.

---

## 12. Confidence and uncertainty

Do not assume that all VLMs expose calibrated token probabilities.

Implement confidence estimation as a modular component with fallbacks.

Possible signals:

1. normalized sequence log probability, if exposed;
2. minimum or average answer-token probability;
3. self-reported structured confidence;
4. agreement across cheap initial candidate sets;
5. answer frequency;
6. response stability under prompt paraphrase.

The default should prefer model probabilities if available and otherwise use agreement-based confidence.

Represent:

```python
confidence in [0, 1]
uncertainty = 1 - confidence
```

Document the limitations of every proxy.

---

## 13. View diversity and overlap

The cache must not count near-duplicate view sets as independent evidence.

### 13.1 Within-set diversity

Use cdViews spatial overlap data where available.

Possible definition:

```python
diversity(view_set) =
    1 - mean_pairwise_overlap(view_set)
```

Clamp to `[0, 1]`.

### 13.2 Between-set overlap

For candidate sets \(A\) and \(B\), support:

```python
jaccard_overlap = |A ∩ B| / |A ∪ B|
```

If geometric overlap is available, prefer a geometry-aware score.

### 13.3 Effective evidence weight

Implement a configurable weight such as:

\[
w_i =
c_i^{\alpha}
d_i^{\beta}
s_i^{\eta}
(1-o_i)^{\gamma}
\]

where:

- \(c_i\): confidence;
- \(d_i\): diversity;
- \(s_i\): mean cdViews criticality score;
- \(o_i\): redundancy/overlap;
- \(\alpha,\beta,\eta,\gamma\): configurable exponents.

Avoid division by zero and clamp all terms.

---

## 14. TDA-style answer cache

Implement a fixed-size cache inspired by TDA, adapted for open-ended QA.

Required behavior:

- reset per scene-question pair by default;
- accept/reject entries according to confidence threshold;
- store the question-conditioned feature;
- store the normalized answer cluster;
- prune low-value or redundant entries;
- expose answer-level aggregate scores;
- optionally maintain a negative cache.

Suggested interface:

```python
class AnswerCache:
    def update(self, entry: CacheEntry) -> bool:
        ...

    def score_answers(
        self,
        query_key: Tensor | None = None,
    ) -> dict[str, float]:
        ...

    def prune(self) -> None:
        ...

    def clear(self) -> None:
        ...
```

### 14.1 Base cache score

For answer \(a\):

\[
S_{\mathrm{cache}}(a)
=
\sum_{i:y_i=a}
w_i
\]

### 14.2 Similarity-aware score

When features are enabled:

\[
S_{\mathrm{cache}}(a \mid q)
=
\sum_{i:y_i=a}
w_i
\exp(\tau \cdot \cos(q,k_i))
\]

where:

- \(q\): aggregate query/context feature;
- \(k_i\): cache key;
- \(\tau\): temperature.

### 14.3 Redundancy-aware pruning

Prefer retaining:

- high-confidence entries;
- geometrically diverse entries;
- entries that are not near-duplicates of existing entries;
- entries representing competing plausible answers when useful for uncertainty estimation.

---

## 15. Optional negative cache

Implement behind a configuration flag.

The negative cache may store low-confidence or contradicted answer hypotheses.

Possible use:

\[
S(a) \leftarrow S(a) - \lambda_{\mathrm{neg}} S_{\mathrm{neg}}(a)
\]

Do not enable it in the default MVP configuration.

Add tests showing it cannot produce NaNs or unbounded negative scores.

---

## 16. Uni-Adapter-style answer prototypes

Implement one prototype per semantic answer cluster.

For answer cluster \(a\), maintain:

```python
prototype[a] = EMA of accepted keys supporting a
```

Update:

\[
p_a \leftarrow
m p_a + (1-m)k_i
\]

followed by normalization.

Required interface:

```python
class PrototypeCache:
    def update(
        self,
        answer_cluster: str,
        key: Tensor,
        weight: float,
    ) -> None:
        ...

    def score(
        self,
        query_key: Tensor,
    ) -> dict[str, float]:
        ...

    def clear(self) -> None:
        ...
```

Prototype score:

\[
S_{\mathrm{proto}}(a)
=
\cos(q,p_a)
\cdot
\log(1+n_a)
\cdot
\bar{w}_a
\]

where:

- \(n_a\): number of supporting entries;
- \(\bar{w}_a\): mean evidence weight.

Graph smoothing is optional and must not be part of the MVP.

---

## 17. Query key for fusion

For a question-local cache, candidate entries are already conditioned on the same question. A query key is still useful for determining which view-set evidence is closest to the aggregate context.

Support:

```python
query_key = normalized weighted mean of accepted candidate keys
```

Weights should use confidence and diversity.

If no valid features exist, fall back gracefully to answer-level cache scoring.

---

## 18. Fusion

Implement a general fusion scorer.

For answer \(a\):

\[
S(a)
=
\lambda_{\mathrm{base}}S_{\mathrm{base}}(a)
+
\lambda_{\mathrm{cache}}S_{\mathrm{cache}}(a)
+
\lambda_{\mathrm{proto}}S_{\mathrm{proto}}(a)
-
\lambda_{\mathrm{instability}}I(a)
\]

Where:

- `S_base`: support for the original cdViews answer;
- `S_cache`: weighted cache support;
- `S_proto`: prototype similarity;
- `I(a)`: instability penalty.

For open-ended output, the base model may provide only one answer. In that case:

```python
S_base(base_answer) = base_confidence
S_base(other_answers) = 0
```

### 18.1 Instability

Possible definition:

\[
I(a) = 1 - \frac{\sum_{i:y_i=a}w_i}{\sum_i w_i}
\]

Also support perturbation-specific instability:

- answer changes under nearby-frame jitter;
- answer changes under leave-one-out variants;
- answer changes under prompt type.

### 18.2 Conservative fallback

If cache uncertainty is too high, return the original cdViews answer.

For normalized answer distribution \(p(a)\):

\[
H(p) = -\sum_a p(a)\log p(a)
\]

If normalized entropy exceeds a threshold, do not override the base answer.

### 18.3 Tie handling

Use deterministic tie-breaking:

1. higher base support;
2. higher cache support;
3. higher prototype support;
4. lexicographic order only as a final deterministic fallback.

---

## 19. Uncertainty gating

Implement a cheap pre-adaptation gate.

Run a small initial set, for example:

- cdViews default;
- top critical;
- diversity-heavy.

If normalized answers agree above a configured threshold, return the consensus or base answer without running all candidate variants.

Example:

```python
if agreement(initial_answers) >= gate_threshold:
    return early_answer
else:
    return full_tta()
```

Log whether each example was adapted.

The configuration must expose:

```yaml
gating:
  enabled: true
  initial_num_sets: 3
  agreement_threshold: 0.67
  entropy_threshold: 0.75
```

---

## 20. MVP pipeline

Implement this first:

```python
def run_mvp(example, cfg):
    prepared = cdviews_adapter.prepare_example(example)

    selection = cdviews_adapter.select_views(
        scene=prepared.scene,
        question=prepared.question,
    )

    base_views = selection.nms_view_ids[: cfg.views_per_set]
    base_response = vlm_runner.answer(
        prepared.question,
        base_views,
        prompt_type=cfg.prompt_type,
    )

    candidate_sets = candidate_set_generator.generate(
        selection=selection,
        scene=prepared.scene,
        question=prepared.question,
    )

    candidate_answers = []
    for candidate_set in candidate_sets:
        response = vlm_runner.answer(
            prepared.question,
            candidate_set.view_ids,
            prompt_type=cfg.prompt_type,
        )

        candidate_answers.append(
            build_candidate_answer(
                response=response,
                candidate_set=candidate_set,
            )
        )

    if gating_allows_early_exit(candidate_answers, cfg):
        return early_exit_result(...)

    cache = AnswerCache(cfg.cache)
    for candidate in candidate_answers:
        cache.update(to_cache_entry(candidate))

    cache_scores = cache.score_answers()
    final_answer = fusion_scorer.fuse(
        base_answer=base_response.answer,
        base_confidence=base_response.confidence,
        cache_scores=cache_scores,
        prototype_scores=None,
        candidate_answers=candidate_answers,
    )

    return build_result(...)
```

The MVP must work without feature embeddings or prototypes.

---

## 21. Full pipeline

After the MVP passes tests, add:

- frozen image/text features;
- question-conditioned keys;
- prototype cache;
- similarity-aware cache scoring;
- entropy-aware fusion;
- adaptive gating.

The full pipeline must gracefully degrade when:

- the encoder is unavailable;
- a view cannot be loaded;
- the VLM returns malformed JSON;
- no candidate passes the confidence threshold;
- all candidate answers are unique;
- all features are identical;
- only one valid candidate view set exists.

In these cases, fall back to the base cdViews answer and log the reason.

---

## 22. Configuration

Use a typed configuration system.

Example full configuration:

```yaml
seed: 42
device: cuda

dataset:
  name: sqa3d
  split: val
  limit: null

cdviews:
  views_per_set: 4
  preserve_original_prompt: true

candidate_sets:
  num_sets: 5
  views_per_set: 4
  include_default: true
  include_top_critical: true
  include_diversity_heavy: true
  include_leave_one_out: true
  include_jitter: true
  include_random_control: false

answers:
  equivalence_mode: embedding_cluster
  embedding_similarity_threshold: 0.88
  canonicalize_spatial_terms: true

features:
  enabled: true
  encoder: siglip
  pooling: score_weighted_mean
  conditioning: concatenation
  normalize: true

confidence:
  source_priority:
    - token_probability
    - agreement
    - structured_self_report
  minimum_cache_confidence: 0.35

cache:
  enabled: true
  max_size: 16
  similarity_temperature: 10.0
  use_negative_cache: false
  pruning: confidence_diversity

prototype:
  enabled: true
  momentum: 0.8
  minimum_support: 2

weights:
  confidence_exponent: 1.0
  diversity_exponent: 1.0
  criticality_exponent: 1.0
  overlap_exponent: 1.0

fusion:
  lambda_base: 1.0
  lambda_cache: 1.0
  lambda_prototype: 0.5
  lambda_instability: 0.5
  fallback_entropy_threshold: 0.8

gating:
  enabled: true
  initial_num_sets: 3
  agreement_threshold: 0.67

logging:
  output_dir: outputs/full_tta3dcache
  save_raw_responses: true
  save_features: false
```

Validate all ranges.

---

## 23. Logging schema

Write one JSONL record per question.

Required fields:

```json
{
  "run_id": "...",
  "scene_id": "...",
  "question_id": "...",
  "question": "...",
  "ground_truth_answers": ["..."],
  "base_views": ["..."],
  "base_raw_answer": "...",
  "base_normalized_answer": "...",
  "base_confidence": 0.0,
  "candidate_sets": [
    {
      "candidate_id": "...",
      "generation_type": "...",
      "view_ids": ["..."],
      "diversity_score": 0.0,
      "overlap_with_base": 0.0,
      "cdviews_scores": [0.0]
    }
  ],
  "candidate_answers": [
    {
      "candidate_id": "...",
      "raw_answer": "...",
      "normalized_answer": "...",
      "semantic_cluster_id": "...",
      "confidence": 0.0,
      "uncertainty": 0.0,
      "diversity_score": 0.0,
      "overlap_score": 0.0,
      "cdviews_score": 0.0,
      "cache_weight": 0.0,
      "prompt_type": "..."
    }
  ],
  "cache_scores": {},
  "prototype_scores": {},
  "final_scores": {},
  "final_answer": "...",
  "adapted": true,
  "fallback_reason": null,
  "num_vlm_calls": 0,
  "runtime_seconds": 0.0,
  "base_correct": false,
  "final_correct": false
}
```

Also create a run manifest with:

- full resolved configuration;
- environment information;
- package versions;
- CUDA information;
- Git commit hashes;
- command line invocation.

---

## 24. Evaluation

Support the original cdViews metrics wherever possible.

At minimum report:

- exact match;
- refined/normalized exact match if the dataset uses it;
- base accuracy;
- adapted accuracy;
- delta;
- base-to-correct correction rate;
- correct-to-wrong regression rate;
- percentage of examples adapted;
- average VLM calls;
- mean runtime;
- accuracy versus VLM-call budget.

Include paired bootstrap confidence intervals for the difference between base and adapted accuracy.

Implement ablation-friendly aggregation by:

- question type;
- scene;
- candidate disagreement;
- base confidence;
- number of views;
- spatial relation questions;
- answer length.

---

## 25. Required baselines and ablations

Implement configuration presets for:

1. **cdViews baseline**
   - one default view set;
   - no cache.

2. **Random extra views**
   - equal inference budget;
   - random candidate sets.

3. **Majority vote**
   - normalized answer frequency only.

4. **Confidence-weighted vote**
   - answer frequency weighted by confidence.

5. **Diversity-weighted vote**
   - confidence and diversity;
   - overlap penalty.

6. **TDA-style answer cache**
   - fixed-size cache;
   - no features required.

7. **Feature cache**
   - similarity-aware keys;
   - no prototypes.

8. **Prototype cache**
   - answer-conditioned prototypes.

9. **Full TTA3DCache**
   - cache;
   - prototypes;
   - uncertainty gating;
   - conservative fallback.

The implementation should make it straightforward to compare majority voting against the full method at an identical number of VLM calls.

---

## 26. Tests

Write unit tests before or alongside each module.

Required tests:

### Answer normalization

- equivalent spatial phrasings group together;
- opposite relations remain distinct;
- articles and punctuation are removed safely;
- number normalization behaves predictably.

### Candidate view sets

- deterministic under fixed seed;
- no duplicate view sets;
- default set is preserved;
- view-set size constraints hold;
- leave-one-out changes exactly one intended element where possible.

### Cache

- rejects low-confidence entries;
- respects max size;
- aggregates equivalent answers;
- overlap lowers effective weight;
- empty cache returns empty scores;
- clear resets state.

### Prototype cache

- normalized prototypes;
- deterministic EMA;
- minimum support behavior;
- no NaNs for zero or degenerate features.

### Fusion

- majority-vote mode matches hand-computed results;
- diversity weighting can prevent duplicate-view domination;
- high entropy triggers fallback;
- ties are deterministic;
- base answer remains unchanged when cache is empty.

### Smoke test

Create a mock cdViews adapter and mock VLM so the complete pipeline runs on CPU without external weights.

The smoke test must include a case where:

- three highly overlapping candidate sets predict the same wrong answer;
- two diverse candidate sets predict the correct answer;
- naive majority vote is wrong;
- diversity/overlap-aware TTA3DCache selects the correct answer.

This test directly demonstrates the intended contribution.

---

## 27. CLI requirements

Support commands such as:

```bash
python scripts/run_cdviews_baseline.py \
  --config configs/baseline.yaml \
  dataset.limit=300

python scripts/run_tta3dcache.py \
  --config configs/full_tta3dcache.yaml \
  dataset.name=sqa3d \
  dataset.split=val \
  dataset.limit=300

python scripts/evaluate_predictions.py \
  --predictions outputs/full_tta3dcache/predictions.jsonl \
  --compare outputs/baseline/predictions.jsonl

python scripts/inspect_run.py \
  --predictions outputs/full_tta3dcache/predictions.jsonl \
  --show-corrections \
  --limit 20
```

Use clear error messages and nonzero exit codes for invalid configurations.

---

## 28. README requirements

Document:

- research motivation;
- relationship to cdViews, TDA, and Uni-Adapter;
- architecture diagram in Mermaid;
- setup with submodules;
- exact reproduction commands;
- MVP commands;
- full method commands;
- ablation commands;
- output format;
- expected failure modes;
- how to add another dataset or VLM;
- limitations.

Include a warning that the current method is training-free and question-local by default.

---

## 29. Implementation order

Work in this order.

### Phase 1: Repository inspection and baseline

- inspect all submodules;
- create `docs/submodule_mapping.md`;
- expose a stable cdViews adapter;
- reproduce the unmodified baseline;
- log all intermediate selected views.

### Phase 2: Candidate view sets and majority vote

- candidate generation;
- VLM wrapper;
- normalization;
- majority-vote baseline;
- deterministic logs.

### Phase 3: Weighted answer cache

- confidence;
- diversity;
- overlap;
- TDA-style cache;
- conservative fusion.

### Phase 4: Feature keys and prototypes

- frozen feature encoder;
- question conditioning;
- similarity-aware cache;
- Uni-Adapter-style answer prototypes.

### Phase 5: Gating and evaluation

- uncertainty gate;
- fallback policy;
- efficiency metrics;
- bootstrap confidence intervals;
- ablation configs.

Do not begin Phase 4 until the MVP in Phase 3 passes tests and can run end-to-end.

---

## 30. Coding standards

- Python 3.11.
- Type annotations on public functions.
- Dataclasses or validated models for structured records.
- Clear docstrings explaining mathematical roles.
- No silent exception swallowing.
- No global mutable cache state.
- Deterministic behavior under a fixed seed.
- Device-safe tensor handling.
- No hard-coded absolute paths.
- Use dependency injection for the VLM and feature encoder.
- Keep external model loading out of unit tests.
- Prefer small pure functions for scoring logic.
- Use structured logging rather than scattered `print` calls.
- Preserve compatibility with the submodules’ licenses.

---

## 31. Scientific guardrails

Do not claim state of the art in code comments or documentation.

Do not assume that more VLM calls constitute adaptation by themselves.

The method is only scientifically meaningful if it can separate:

- gains from additional inference;
- gains from answer self-consistency;
- gains from view diversity;
- gains from cache retrieval;
- gains from prototypes;
- gains from uncertainty gating.

The main comparison is:

```text
full TTA3DCache vs. majority vote at equal VLM-call budget
```

Log sufficient data to analyze both corrections and regressions.

---

## 32. Deliverables

Produce:

1. a working baseline wrapper around cdViews;
2. a working MVP answer-cache pipeline;
3. a full feature/prototype pipeline;
4. all required tests;
5. all configuration presets;
6. CLI scripts;
7. `docs/submodule_mapping.md`;
8. a complete README;
9. example output from the mock smoke test;
10. a concise implementation report in:

```text
docs/implementation_report.md
```

The implementation report must include:

- files changed;
- design decisions;
- deviations from this specification;
- unresolved issues;
- commands used for tests;
- test results;
- next recommended experiment.

---

## 33. Expected first end-to-end milestone

The first milestone is not full benchmark performance.

It is a CPU-compatible mock demonstration where:

1. cdViews provides a ranked set of mock views;
2. five candidate view sets are generated;
3. a mock frozen VLM returns controlled answers;
4. naive majority voting fails because three wrong candidates are redundant;
5. TTA3DCache downweights the overlapping candidates;
6. the final adapted answer is correct;
7. the complete JSONL record is written;
8. all tests pass.

After this milestone, integrate the real cdViews inference path on a small validation subset.

Begin by inspecting the repository and writing `docs/submodule_mapping.md`. Then implement Phase 1 and proceed sequentially. Do not skip baseline reproduction, testing, or logging.
