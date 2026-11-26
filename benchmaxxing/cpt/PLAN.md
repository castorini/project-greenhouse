# Iteration 1: continued pretraining on the evaluation corpora

**Status (2026-11-26):** the plan, data preparation, training, evaluation and export scripts are ready, and data
preparation and the training pipeline pass CPU smoke tests (`tests/`).

- **3B: done.** The reranker fine-tuned on the CPT checkpoint (section 6: run 1's recipe and training plan, final
  checkpoint) scores DL 0.6384 and BEIR 0.5476 (nDCG@10 means; corpus-aware, see section 7), against run 1's 0.6414
  and 0.5438: -0.003 and +0.004, both under the 0.005 the rules treat as noise. Before fine-tuning, the CPT model
  scores CORE 0.4498 (base 0.4502, every example), MMLU 0.4899 (0.4927) and MMLU-Pro 0.1718 (0.1776).
- **0.6B:** the CPT run can start now: `configs/train_0.6b.yaml`, from the released
  [`castorini/gaggle-nanochat-0.6b-pretrained-climbmix-20261121`](https://huggingface.co/castorini/gaggle-nanochat-0.6b-pretrained-climbmix-20261121), with
  [`castorini/gaggle-reranker-0.6b-20261122`](https://huggingface.co/castorini/gaggle-reranker-0.6b-20261122) as its reranker baseline.

**Question.** If a Gaggle base model keeps pre-training on the raw documents of the collections the reranker is
evaluated on, does the same reranker fine-tuning then score higher?

**Labelling.** These models see the evaluation corpora (never the queries or judgements). Their BEIR results are
**corpus-aware, not zero-shot**, and must be labelled that way wherever they are reported. TREC DL is in-domain for
every Gaggle reranker anyway, since its fine-tuning data (RLHN-250K) is built on MS MARCO. With `exclude_ood: true`
the four collections the paper counts as out of domain (TREC-COVID, NFCorpus, Robust04, TREC-News) stay out of the
mix, so their results are not corpus-aware.

---

## 1. Decisions

| # | Question | Default | Section |
|---|---|---|---|
| 1 | When to mix the corpus in | **After pre-training (CPT)**, from the released Gaggle base models: they are the only checkpoints available. Within the run, corpus and replay are interleaved from start to end. | 2 |
| 2 | What to mix in | **Raw document text** of all the evaluation corpora (all seven BEIR collections, Robust04 and TREC-News included; MS MARCO v1 and v2), **balanced** between BEIR, MS MARCO v1 and MS MARCO v2 (1 : 1 : 1). No queries, no judgements, no prompt/answer format. | 3 |
| 3 | Replay | **25% of training tokens**: pre-training data (ClimbMix), interleaved at a constant ratio, added on top of the corpus. | 4 |
| 4 | Models | **Gaggle 3B** (`castorini/gaggle-nanochat-pretrained-climbmix-20260924`) and **Gaggle 0.6B** (`castorini/gaggle-nanochat-0.6b-pretrained-climbmix-20261121`). | 6 |
| 5 | Learning rate | Peak **0.1 × the original run's peak**, 1% warmup, linear decay to 0 over the second half. | 5 |
| 6 | Reranker stage | Unchanged: the paper's run-1 recipe for the 3B model; for the 0.6B model, the recipe of its released reranker (run 1's, with a pinned 120,000-token budget on one H200). Only the starting weights change. | 6 |

Everything above is a default in one of two files: the mix in `configs/data.yaml`, the training run in
`configs/train.yaml`. Any key can be changed in the file or on the command line (`--set KEY=VALUE`).

---

## 2. When to mix

**After pre-training.** The released Gaggle checkpoints are the ends of their runs (the 3B model finished at about
0.07 × its peak learning rate), and no earlier checkpoint is released. So the corpus enters after pre-training:
continued pretraining (CPT) from the released weights, with its own short warmup and decay.
- **It is affordable.** The default mix is about 4.4B tokens, about 1.5% of the 287B of pre-training.
- **It isolates the question.** The base model and the reranker recipe stay fixed; only the CPT stage is added.

**Within the CPT run**, corpus and replay documents are interleaved from the first step to the last, at a constant
replay share (`replay.schedule: constant`; `front_loaded` / `back_loaded` with `replay.skew` are also supported).

---

## 3. What to mix

**Raw documents of the evaluation corpora, and nothing else.** Every document is used as plain text with the
language-modelling loss on every token:
- **Format:** `title\n\ntext`, or the text alone when there is no title. MS MARCO passages have no title.
- **Length:** documents over 3,500 characters (about 800 tokens) are split at a paragraph, then sentence, then word
  boundary.
- **Exact duplicates** are dropped.
- **Never used:** queries, relevance judgements, BM25 candidate lists, or any selection by query.

The 12 test sets read 9 corpora, in three groups:

| Group | Corpus | Test sets | Tokens (one pass) | Most passes |
|---|---|---|---:|---:|
| BEIR | TREC-COVID, NFCorpus, FiQA, SciDocs, SciFact | the same names | ~60M together | 4 |
| BEIR | Robust04, TREC-News | the same names | ~851M together | 1 |
| MS MARCO v1 | MS MARCO v1 passages | DL19, DL20 | ~715M | 2 |
| MS MARCO v2 | MS MARCO v2 passages | DL21, DL22, DL23 | ~8.2B | 1 |

**Balance.** The groups enter at the ratios in `groups:`, 1 : 1 : 1. The group that runs out first, at its sources'
most passes, sets the corpus size; every other group contributes its ratio of that. Below one pass a source
contributes a fixed random subset of its documents (by hash, independently of any query). The default mix:

| Group | In the mix | Of what is available |
|---|---:|---|
| BEIR | ~1.09B | everything: the open corpora 4 times, Robust04 and TREC-News once (BEIR sets the size) |
| MS MARCO v1 | ~1.09B | about 1.5 passes: every passage once, half of them twice |
| MS MARCO v2 | ~1.09B | about 13% of the passages |
| **Corpus** | **~3.3B** | with 25% replay: **~4.4B tokens**, about 4,000 optimizer steps of 2^20 tokens |

Notes:
- **Why balance.** Taken whole, MS MARCO v2 alone would be about 90% of the corpus and BEIR about 2%.
- **MS MARCO v2 exposure is partial.** About 13% of its passages, and so about 13% of the DL21–23 candidates, are
  in the mix.
- **The small BEIR corpora get 4 passes.** At one pass NFCorpus would be a negligible share. Muennighoff et al. (2023)
  find that up to about 4 passes over repeated data are nearly as useful as fresh data. The passes follow one
  another, each spread evenly over its own part of the run and in its own order.
- **Robust04 and TREC-News are always included.** They are read from Pyserini's prebuilt BEIR indexes, the same ones
  the evaluation uses (needs `pyserini` and Java 21). Their text is licensed by NIST and may not be redistributed:
  never publish a built mix, and check the licence before releasing a model trained on one.
- **Out-of-domain collections can be kept out.** The paper counts four BEIR collections as out of domain for
  RLHN-250K: TREC-COVID, NFCorpus, Robust04 and TREC-News (marked `ood: true`). `exclude_ood: true` leaves them out
  of the mix entirely, and the mix is then sized with the ratios in `groups_exclude_ood:` (1 : 4 : 4, because
  in-domain BEIR is only ~18M tokens): BEIR ~72M, MS MARCO v1 and v2 ~0.29B each, ~0.86B tokens with replay.

---

## 4. Replay: how much, how, and from where

**Answer: 25% of the CPT tokens are pre-training data, mixed in at a constant ratio, document by document, on top of
the corpus.**

### What the literature says

| Work | Setting | Finding on replay or learning rate |
|---|---|---|
| Ibrahim et al. 2024 ([2403.08763](https://arxiv.org/abs/2403.08763)) | CPT of 405M–10B models, LR re-warmed and re-decayed | **5%** replay enough for a weak shift (web → web); **25%** for a strong one (English → German). A higher re-warmed LR adapts more and forgets more. |
| Gupta et al. 2023 ([2308.04014](https://arxiv.org/abs/2308.04014)) | re-warming for CPT | Re-warming raises the loss on the old data at first; the peak LR trades adaptation against forgetting. |
| Guo et al. 2024 ([2406.14833](https://arxiv.org/abs/2406.14833)) | the "stability gap" in CPT | Performance first drops, then recovers. A mixture close to the pre-training data shrinks the drop. |
| Parmar et al. 2024 ([2407.07263](https://arxiv.org/abs/2407.07263)) | a recipe for continued pretraining | A general blend first, then a target-focused blend switched in late in the decay (at about ηmax/5). |
| Bethune et al. 2025 ([2502.06042](https://arxiv.org/abs/2502.06042)) | scaling laws for forgetting with pre-training data injected | Even **1%** of pre-training data limits forgetting substantially; smaller models forget more. |
| EvoLM, 2025 ([2506.16029](https://arxiv.org/abs/2506.16029)) | CPT then SFT, measured after SFT | A moderate replay share (**16%** in their sweep) gave the best results after fine-tuning. |
| Abbes et al. 2025 ([2508.01908](https://arxiv.org/abs/2508.01908)) | replay vs. model size in CPT | Replay around **25%** is a compute-efficient way to limit forgetting. |
| D-CPT Law ([2406.01375](https://arxiv.org/abs/2406.01375)); CMR ([2407.17467](https://arxiv.org/abs/2407.17467)) | scaling laws for the domain/general ratio | The best ratio depends on the model and the domain, and can be predicted from small runs. |
| Domain CPT practice | Code Llama, Llemma, DeepSeekMath, PMC-LLaMA, Meditron | General text at about **1–10%** of the domain mix (Code Llama 7%, DeepSeekMath 10% Common Crawl, Meditron about 1%). |
| Corpus-aware retrieval pretraining | coCondenser, COCO-DR, GPL, Llama2Vec | Unsupervised training on the target corpus improves retrieval on that corpus: the precedent for this project. |

None of these works uses Muon, which trains the Gaggle models' matrix weights.

### Why 25%, and why constant

- **Our shift sits between the two cases of Ibrahim et al.** It stays in English but is narrower text: web passages,
  newswire, biomedical, scientific and financial documents. That puts it between their 5% and 25%. We take the safe
  end because the reranker needs general knowledge (BEIR rewards it), so forgetting costs the final metric, not just
  a language-modelling benchmark.
- **Ours is true replay.** The replay comes from ClimbMix shards the 3B model was trained on (shards 0–5899; its
  data loaders were past shard 5,949 at release). The 0.6B model's loaders were at shard 6,029 at release
  (`dataloader_state_dict` in its `meta_229067.json`), so the same replay shards are seen data for both models.
- **The cost is modest.** 25% replay means a third more tokens than none.
- **Constant ratio, interleaved document by document, fixed when the data is built.** Every stretch of the run sees
  the same share, so a run stopped early or preempted is still the same experiment. The trainer only reads shards in
  order; `manifest.json` records the realised share of every shard.
- **Added on top, not substituted.** The replay share does not change which corpus documents are trained on.

---

## 5. Optimisation

| Setting | Value | Why |
|---|---|---|
| Peak LR | 0.1 × the original run's peak, for all four parameter groups (Muon matrices, embeddings, unembedding, scalars) | The 3B model ended at about 0.07 × peak, so this is a mild re-warm: enough to adapt, little to forget (Ibrahim et al.; Gupta et al.). |
| Schedule | 1% linear warmup, constant, linear decay to 0 over the second half | Pre-training decayed over its last 55%. Decaying to 0 gives a finished model, as in the replay papers. |
| Batch | 2^20 tokens per step (2,048-token sequences) | About 4,000 steps on the default mix. Pre-training ended at 2^21. |
| Weight decay | the original run's Muon amplitude, as is (3B 0.021764; 0.6B 0.112984, its 1M-batch value, held at 2M); cosine to 0 over the run | nanochat would otherwise rescale it for the short horizon (by tens of times). |
| Length | one pass: 95% of the mix's tokens | nanochat's loader would wrap into a second pass, and its packing crops a few percent of tokens. |
| Precision | FP8 matmuls, as in pre-training (`fp8: false` on GPUs older than H100) | |
| Optimizer state | fresh | The released checkpoints carry no optimizer state. |

All of it is in `configs/train.yaml`, as flags of the shared trainer (nanochat's own, in `pretrain/continue/`). The learning rates are written out (the
original run's peaks × 0.1), and nanochat's own batch-size and horizon rescaling is switched off with
`literal-hparams`.

---

## 6. The run

For each model (3B and 0.6B), with the repository's shared tools (README.md has the commands):
1. `prepare_data.py configs/data.yaml`: the mix (once, shared by both models).
2. The continued-pretraining trainer of `pretrain/continue/` with `configs/train.yaml` (3B) or
   `configs/train_0.6b.yaml` (0.6B): CPT.
3. `scripts/base_eval.py --checkpoint-dir` (bpb on the held-out, seen and replay sets; CORE): the language model
   before and after.
4. The paper's reranker fine-tuning and evaluation, with the run's checkpoint directory as the backbone.

**Baselines.** For the 3B model, the paper's run 1 (`castorini/gaggle-reranker-run1-20260926`; DL 0.641, BEIR
0.544). For the 0.6B model, its released reranker (`castorini/gaggle-reranker-0.6b-20261122`; DL 0.634, BEIR 0.533),
trained with `reranker/configs/experiments/gaggle06b_lce.yaml`, whose pinned budget and packing the CPT reranker
reuses, so both see the same batches.

**Cost (extrapolated from pre-training throughput).** The default mix is about 4.4B tokens, about 1.5% of a
pre-training run: about 25 H100-hours for the 3B model (pre-training took ~1,600 for 287B tokens), and about 7 for
the 0.6B model (~1.4M tokens/s on 8 H100s in pre-training, about 450 H100-hours for 287B tokens).

**Rules:**
- Pick no checkpoint by test-set score. The reranker uses its final checkpoint, and the CPT run is used at its final
  step.
- Judge reranker differences against run-to-run noise: four trainings of the paper's recipe span 0.638–0.641 on the
  DL mean and 0.542–0.545 on the BEIR mean. Treat a difference under 0.005 on a mean as noise.
- Report the BEIR mean over all 7 collections, as the paper does.

---

## 7. What to report

For each model:
- **From `scripts/base_eval.py`:** held-out corpus bpb, seen-corpus bpb and their difference (memorisation),
  held-out ClimbMix bpb, and CORE, before and after CPT.
- **From `eval/results/`:** nDCG@10 per collection, the DL mean and the BEIR mean, next to the baseline.

Label every reranker number trained from a CPT model as corpus-aware, except, under `exclude_ood`, the four
out-of-domain collections.

## 8. Risks

- **No gain.** The reranker fine-tune (one epoch of 250K query groups) may wash out what CPT adds. The
  memorisation measure (seen vs. held-out bpb) tells "CPT did not take" apart from "it took but did not help".
- **Forgetting hurts BEIR more than exposure helps.** The held-out ClimbMix bpb and CORE show it.
- **Robust04 and TREC-News dominate the BEIR group** (~79% of its tokens), so the five smaller collections get a
  small share of the mix.
- **Muon is untested in the CPT literature.** The learning rate and weight decay follow the papers' AdamW findings.

## References

- Ibrahim et al. 2024. Simple and Scalable Strategies to Continually Pre-train Large Language Models. arXiv:2403.08763.
- Gupta et al. 2023. Continual Pre-Training of Large Language Models: How to (re)warm your model? arXiv:2308.04014.
- Guo et al. 2024. Efficient Continual Pre-training by Mitigating the Stability Gap. arXiv:2406.14833.
- Parmar et al. 2024. Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models. arXiv:2407.07263.
- Bethune et al. 2025. Scaling Laws for Forgetting during Finetuning with Pretraining Data Injection. arXiv:2502.06042.
- EvoLM, 2025. arXiv:2506.16029.
- Abbes et al. 2025. Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models.
  arXiv:2508.01908.
- Que et al. 2024. D-CPT Law: Domain-specific Continual Pre-Training Scaling Law. arXiv:2406.01375.
- Gu et al. 2024. CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training. arXiv:2407.17467.
- Muennighoff et al. 2023. Scaling Data-Constrained Language Models. arXiv:2305.16264.
- Rozière et al. 2023 (Code Llama), Azerbayev et al. 2023 (Llemma), Shao et al. 2024 (DeepSeekMath), Wu et al. 2023
  (PMC-LLaMA), Chen et al. 2023 (Meditron).
- Gao and Callan 2022 (coCondenser), Yu et al. 2022 (COCO-DR), Wang et al. 2022 (GPL), Li et al. 2023 (Llama2Vec).
