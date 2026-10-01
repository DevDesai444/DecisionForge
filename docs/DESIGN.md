# DecisionForge design notes (principles · architecture · training method)

## 1. Problem and principle

Jev (the TypeSafe "System One" model; public docs: docs.typesafe.ai) treats an
LLM as a **decision function**: it takes `state + questions` (three atomic
question types — choice / score / noul) and returns a **complete calibrated
probability distribution**, with no autoregressive decoding. Its value lies in
latency (a single forward pass), cost (no output tokens), and the impossibility
of hallucination.

DecisionForge's goal: reproduce this approach with a 0.6B-class open backbone,
and keep **training data format == inference request format == the public Jev
contract**, so that production logs can be fed straight back into training.

Core principle: every (question, candidate) pair forms a "candidate path" that
shares the state+question prefix and differs only in the candidate segment; take
the hidden state of the path's final token, pass it through a shared scalar head
to obtain a logit; a softmax over the K logits of the same question yields the
full distribution. Training is therefore **probability learning** (proper
scoring), not generative SFT.

## 2. Architecture choices (and why)

| Decision | Choice | Rationale |
|---|---|---|
| backbone | Qwen3-0.6B (base) | strongest open small model within the latency/cost budget; 0.6B allows full-parameter training on a single GPU |
| read-out | shared prefix + per-candidate scalar head (2-layer MLP) | matches the public Jev semantics: candidates are independent, parallel, and K can reach 255; questions are naturally isolated (no cross-attention) |
| alternative read-out (not adopted) | concatenate all candidates into one context (listwise) | models candidate competition, but the context grows with K, is order-sensitive, and violates the Score "evaluate each level independently" semantics; kept for a v2 ablation |
| loss | CE + 0.5·Brier (over the full candidate set per question) | CE on soft targets is equivalent to forward KL; Brier directly optimises calibration; both are proper scoring rules |
| calibration | grid-fit a scalar temperature on the calibration split | a common prototype gap: a default of T=1 is not calibrated; we fit it explicitly and write it into the checkpoint |
| memory | gradient checkpointing (use_reentrant=False) + SDPA (flash backend) | see the measurements below; switching to `flash_attention_2` only needs a change to attn_implementation |

## 2.5 Memory and attention backend (measured)

For a 0.6B model, a 64×600-token training micro-batch with gradient
checkpointing + bf16 autocast, `decisionforge.bench_attention` measured on an
RTX 4080 SUPER 32G:

| attn_implementation | peak memory | time per step |
|---|---:|---:|
| sdpa (built-in flash kernel, default) | 8.53 GB | 3553 ms |
| eager (maths reference implementation) | 12.45 GB | 5444 ms |
| flash_attention_2 (flash-attn 2.8.3, measured) | 8.53 GB | 3554 ms |

Conclusion: the first lever against OOM is gradient checkpointing
(31GB→~7GB); the second is SDPA's flash backend (31% less memory and 53% faster
than eager — shipped with torch≥2.2, no extra install needed). **In practice
flash-attn 2.8.3 (FA2) ties exactly with the built-in SDPA-flash**
(8.53GB/3554ms), so at the 0.6B + checkpointing scale no extra install is
required; scale up the model or sequence and then switch to
`attn_implementation="flash_attention_2"` — training and inference share the
same switch.

## 3. Data strategy: labels first, synthesis to fill gaps

Data source: `LangAGI-Lab/Mind2Web-axtree-cleaned-lite` (the axtree-cleaned
version of Mind2Web, a public dataset). Each annotated step carries its own
**pos/neg candidate elements** — this is ready-made supervision, used directly:

- `action` (choice): gold = a uniform distribution over the positive candidates
  (`uniform_over_positive_elements`); negative candidates are randomly sampled
  up to the cap (default 12, K≥2).
- `is_target` (noul): draw a random candidate to form a self-contained
  proposition; the truth value is whether it is a positive.
- **Splits are partitioned by a discrete hash of the website, and a website
  never crosses a split** (leakage guard; the builder asserts this).
- state is a **canonical JSON string** (sorted keys), fixing the non-canonical
  dict serialisation common in prototypes.

Augmentation only fills the two kinds of supervision the source data lacks:

1. `difficulty` (score, a 3-level rubric): generate and normalise a difficulty
   distribution (`generated_score_distribution`).
2. train-set `action` augmentation: `(1-λ)·gold + λ·generated` (λ=0.25), adding
   uncertainty to sparse labels; dev/test keep their original labels for
   unbiased evaluation.

Generated results go into an append-only cache (auditable, resumable,
reproducible).

## 4. Training pipeline

```
build_web.py  → records (train/dev/calibration/test/ood, websites disjoint)
synthesize.py → +difficulty score targets; train action augmentation (generation cache)
train.py      → head-only warm-up (12 steps) → full-parameter (600 steps, bf16, grad-clip 1.0)
                AdamW: backbone 2e-5 / head 2e-4; dev CE selection every 50 steps
                fit the temperature on the calibration split at the end
predict.py    → Jev-contract answers (choice/noul/score + confidence)
serve.py      → POST /v1/systemone; the request log (JSONL) is future training data
evaluate.py   → top1-in-positives / Brier / ECE / MAE / per-website breakdown
```

## 5. Relationship to NanoJev (TianyuCodings)

DecisionForge is an **independent implementation**: it copies none of that
project's code or git history. The public material referenced: the TypeSafe
public API docs (the interface contract), the public Mind2Web dataset, and the
**public conclusions** of community projects (NanoJev / TheoLeeCj-decisionforge)
— for example "the training domain does not match the real serving domain",
"the prototype did no temperature calibration", "candidate paths re-encode the
prefix". These are design inputs, not sources of code. Architecture differences
are in the table above; the main empirical difference: the training domain moves
from a home-grown game to real web pages.

## 6. Acceptance criteria

1. `python -m decisionforge.train` runs end to end, dev CE converges, and the
   checkpoint reloads.
2. On test/ood: choice top1-in-positives is clearly above the uniform baseline;
   noul Brier/ECE are reported; difficulty MAE is reported; the per-website
   breakdown shows no abnormal leakage.
3. `python -m decisionforge.serve` returns contract-compliant JSON on
   /v1/systemone, with logs written to disk.
4. Sampled manual review: the answer distributions match the task semantics.

## 7. v2 roadmap (after the 2026-09-19 review)

**Already implemented (in v2 training)**: fused AdamW + length bucketing (~2x
throughput measured), cosine LR + warmup, deterministic-target label smoothing
(ε=0.05), per-question-type calibration temperature, full wandb metrics
(loss_ce/loss_brier/grad_norm/lr/sec/peak_mem + dev per-question-type CE,
double-written to local JSONL).

**Dimensions compared against Jev** (beyond accuracy): latency, cost (Jev
$0.042/MTok input vs the marginal cost of local inference ≈ electricity),
calibration (Brier/ECE), contract interoperability (the same /v1/systemone).

**Model v2 candidates**: ① a candidate-competition head — add a lightweight
set-attention on top of the per-candidate scalar head (to model the IIA
limitation), validated by ablation; ② inference-prefix KV reuse (encode
state+question once, fan the candidates out, expected to cut latency
several-fold again); ③ scale the backbone to Qwen3-1.7B (trainable on a 32G
card).

**Data v2 candidates**: K 12→20 + same-label hard negatives; add the task
trajectory (previous-step actions) to state; two nouls per record (one positive,
one negative); align with Mind2Web's official cross-website split for external
comparability.

**Training v2 candidates**: an RLCD-style control arm (sample the full
distribution + proper-scoring reward), per-question-type loss weighting,
early-stopping patience, a direct ECE optimisation term. TypeSafe has published
only the name of RLCD and its calibration objective, not the reward function,
sampler, or optimisation algorithm; this control arm is therefore an independent
research implementation and does not claim to reproduce Jev.

**General-task v2 candidates**: intent routing, tool/API selection,
safety/escalation triage, document decisions, ordered risk scoring. The entry
bar is a finite typed answer, auditable labels or verifiable outcomes, and a
fixable train/dev/test/OOD split — without mixing open-ended text generation
into the decision benchmark.

**Untuned baseline (same-backbone control)**: a control using the raw weights
and the same candidate-scoring protocol.
