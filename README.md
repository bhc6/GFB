# StablePrompt-DCPS — frozen-generator control for RL-based prompt optimization

This directory holds the **StablePrompt audit** from *"Simplicity Goes Far: Auditing
Prompt Optimizers with Demonstration-Conditioned Prompt Search."* It runs a single-variable
ablation inside Kwon et al.'s StablePrompt pipeline: keep everything, remove only the PPO
policy update, and measure what the update was buying.

* **StablePrompt-PPO** — the original pipeline, agent model trained with PPO.
* **StablePrompt-DCPS** — identical scaffolding, **frozen** generator. Demonstration-Conditioned
  Prompt Search: sample `k` demonstrations at random, format them into a meta-prompt, draw
  candidate prompts from the frozen agent model, score them, retain the Top-`K`. No gradient
  update, no optimizer state, no reflection or rewrite loop.

> Naming note: this directory and its `*_gfb.py` files are **legacy naming** ("GFB",
> "Generate & Filter"). The canonical paper name is **StablePrompt-DCPS**, and
> **DCPS = Demonstration-Conditioned Prompt Search**. File names are kept as-is so
> they still match the WandB run history; only the documentation is canonicalized.

## Setup (as reported in the paper)

Target and agent models are both `gemma-1.1-7B-it`, following StablePrompt's full
configuration. Three task families, original scoring protocols:

1. **Text classification** — 6 GLUE/SuperGLUE subsets.
2. **Multi-task QA** — MMLU, 57 subjects.
3. **Instruction induction & generation** — 24 Instruction Induction (II) + 18 Big-Bench
   Instruction Induction (BBII) subsets, split into BBII-TC and BBII-Gen.

Reported scores are the mean over three seeds of the best test score among the Top-5 prompts
ranked by training reward. That protocol is optimistic in absolute terms, but it is applied
identically to PPO and DCPS, so the comparison is unaffected. Two reproduction fixes were
applied to the original code: a train/test overlap in II, and missing batch weights in
StablePrompt's Softmax-difference metric.

Dropping gradients and PPO optimizer state lets DCPS run **inference-only on one A40 48 GB**,
where StablePrompt-PPO needs the original **A100 80 GB**.

## Key results

Macro-averages over subsets; Δ is PPO − DCPS, so positive means PPO is ahead.
Cost is measured hours/USD per run.

| Task family | PPO | DCPS | Δ | 95% CI | Cohen's *d* | DCPS (h/$) | PPO (h/$) | Verdict |
|---|---|---|---|---|---|---|---|---|
| GLUE/SuperGLUE | 76.7 | **77.1** | −0.4 | [−2.30, +1.97] | −0.14 | 4.83 / 1.93 | 3.97 / 5.96 | comparable |
| BBII-TC | **57.6** | 57.2 | +0.4 | [−0.68, +1.63] | +0.20 | 3.38 / 1.35 | 3.17 / 4.76 | comparable |
| BBII-Gen | **63.0** | 60.7 | +2.3 | [+0.15, +4.35] | +0.79 | 2.66 / 1.06 | 1.84 / 2.76 | PPO better |
| II @30 epochs | **48.7** | 44.3 | +4.4 | [+0.15, +8.70] | +0.41 | 2.76 / 1.10 | 6.65 / 9.98 | PPO better |
| II @100 epochs | — | 48.6 | +0.1 | [−2.67, +2.63] | +0.01 | 14.74 / 5.90 | — | gap closed |
| MMLU | **55.9** | 54.1 | +1.8 | [+0.99, +2.74] | +0.53 | 6.58 / 2.63 | 6.38 / 9.57 | PPO better |

What this supports, stated narrowly:

* On template-like classification (GLUE/SuperGLUE, BBII-TC) the PPO update has **limited
  marginal value** — |Δ| ≤ 0.4 pp, CIs straddling zero.
* Where target behavior is harder to reach by random conditioning, **PPO still leads**:
  +2.3 pp on BBII-Gen, +1.8 pp on MMLU.
* On II, extending DCPS to 100 epochs closes the gap against PPO@30 (+0.1 pp). Since PPO was
  not re-run at matching epochs, this suggests extra search budget can **substitute** for the
  policy update; it does **not** show DCPS dominates.
* Where accuracy is matched, the saving is real: $4.76–$5.96 (A100) → $1.35–$1.93 (A40),
  roughly one-third the cost at our rates.

An earlier draft of this README claimed "77.6% vs APPO's 76.4%" and that the control
"outperforms or matches prior RL-based methods." Both are superseded — use the table above.

## Repository layout

PPO runners (StablePrompt-PPO) and their DCPS counterparts share dataset loaders and metrics;
the `_gfb` suffix marks the frozen-generator variant.

| File | Role |
|---|---|
| `tc.py` | StablePrompt-**PPO** — text classification (GLUE/SuperGLUE) |
| `qa.py` | StablePrompt-**PPO** — MMLU (PPO machinery via `utils.py`) |
| `origin_ii.py` | StablePrompt-**PPO** — instruction induction |
| `tc_gfb.py` | StablePrompt-**DCPS** — text classification |
| `qa_gfb.py` | StablePrompt-**DCPS** — MMLU |
| `ii_gfb.py` | StablePrompt-**DCPS** — instruction induction |
| `bbii_tc_gfb.py` | StablePrompt-**DCPS** — BBII-TC (classification subsets) |
| `bbii_tg_gfb.py` | StablePrompt-**DCPS** — BBII-Gen (generation subsets) |
| `utils.py` | Shared helpers incl. the PPO trainer setup (PPO paths only) |
| `ii_utils.py` | Evaluation: `evaluation_ii_batch`, F1 / EM / Exact-Set / contains, prompt formatting |
| `dataset_utils.py` | Dataset loaders (`load_all_dataset`, `load_qa_dataset`, `load_generation_dataset`) |
| `qa_validation.py` | MMLU validation-split helper |
| `automatic_prompt_engineer/` | Vendored APE — retains its upstream LICENSE and attribution |

## How the DCPS path works

1. **Sample** `k` demonstrations at random from the training set and format them into a
   meta-prompt (`FmtMeta`). The demonstration set supplies *only* demonstrations; rewards are
   computed on training batches.
2. **Generate** candidate prompts from the frozen agent model. Sampling is controlled in-code
   via `generation_kwargs` (`top_p`, `top_k`, `max_new_tokens`, `do_sample`).
3. **Evaluate** each candidate with the target model through `ii_utils.evaluation_ii_batch`,
   using the task's original metric.
4. **Select** by validation reward into a bounded Top-`K` queue
   (`TopAccuracyTextsNoDuplicates`); the retained prompts are scored on the held-out test set.

WandB logs run metadata, running metrics, and the final results table.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run a DCPS text-classification job, then the PPO counterpart for comparison:

```bash
python tc_gfb.py --agent_model google/gemma-1.1-7b-it \
                 --target_model google/gemma-1.1-7b-it \
                 --seed 42
python tc.py     --agent_model google/gemma-1.1-7b-it \
                 --target_model google/gemma-1.1-7b-it \
                 --seed 42
```

Each script defines its own flags — check its `argparse` block for the exact names, epoch
counts, and Top-`K` settings rather than assuming they match across files. Use `--cache_dir`
to point at a local HuggingFace cache.

## Reproducing the table

Run every DCPS entrypoint over three seeds and macro-average per task family. II needs both
budgets (30 and 100 epochs) to reproduce the "gap closed" row. PPO rows come from `tc.py`,
`qa.py`, and `origin_ii.py` on A100-class hardware; DCPS rows run inference-only on an A40.

## Citation and attribution

Built on StablePrompt (Kwon et al., 2024) and the vendored APE codebase; both keep their
upstream licenses and attribution. If you use this audit, cite the DCPS paper.

