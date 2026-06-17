# Margin-Adaptive Retrieval Complexity

Code and figures accompanying the paper

> **Margin-Adaptive Retrieval Complexity: A Separation Between Normalized and Max-Plus Attention**
> Alfonso Mulet Vázquez

This repository contains the numerical checks that exercise the paper's theorems on finite
instances, the scripts that generate every figure, and the per-head attention-tail measurements
reported in the average-case section. Everything is CPU-only, dependency-light, and seeded for
exact reproduction.

The repository is a companion to the proofs, not a substitute for them: the contribution of the
paper is mathematical, and these scripts illustrate the closed forms and falsification-test the
arguments on finite instances. A finite computation cannot establish an asymptotic statement; where
a script touches a theorem it does so as a sanity check on concrete instances.

## Layout

```
validate/run_all.py          Numerical PASS/FAIL checks for each theorem/proposition
figures/make_all.py          Regenerates every figure in the paper (300 DPI)
figures/*.pdf                The figures as included in the paper
innovations/                 Additional measurements and theory-motivated experiments
requirements.txt             Pinned dependency versions
```

### `validate/run_all.py`

Runs one check per formal result and prints `PASS` / `FAIL` with the criterion used. Among the
checks: margin-blindness (the output shift equals `w_j·delta·u` to machine precision), the blur
sandwich bound (tested in both directions over many random instances), the correlated-margin law
(`E[Delta]/E[Delta_iid] -> sqrt(1-rho)`), the tail trichotomy (exponential / Gaussian / uniform
laws against simulation), the depth contraction (`gamma^L`), and the Augmented-Indexing adversary
for the decision-version lower bound.

### `figures/make_all.py`

Regenerates the figures: the blur sandwich, the read-ratio signature, the depth collapse, the
average-case speedup curve, the correlation rescaling, and the depth--margin composition. Outputs
go to `figures/`.

### `innovations/`

- `multimodel_tail_analysis.py` — per-head Kolmogorov--Smirnov goodness-of-fit of the attention
  top-gap to exponential / half-Gaussian / uniform families, for an arbitrary Hugging Face causal
  language model. Captures the model's true pre-softmax attention scores.
- `tail_robustness_suite.py` — the same measurement across several prompts and architectures,
  reporting per-model exponential-fit share and its spread across prompts.
- `real_model_tail_analysis.py`, `real_model_tail_robustness.py` — earlier single-model variants.
- `multihead_margin.py` — minimum-over-heads margin and the induced cost bound.
- `margin_estimation.py` — estimating the expected top gap from a small random sample via
  extreme-value theory.
- `margin_eviction.py`, `margin_eviction_realmodel.py` — a theory-motivated KV-cache eviction
  policy, benchmarked on a synthetic stream and on real-model attention against recency,
  heavy-hitter, and random baselines. The real-model script reports an honest mixed result,
  including where the policy does *not* beat the recency baseline.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The tail-measurement scripts download small pretrained models (GPT-2, Pythia-160M, OPT-125M) via
the `transformers` library on first run; the validation and figure scripts need only `numpy`,
`scipy`, and `matplotlib`.

## Running

```bash
python validate/run_all.py                                  # theorem checks
python figures/make_all.py                                  # regenerate figures
python innovations/tail_robustness_suite.py                 # cross-model tail measurement
python innovations/margin_eviction_realmodel.py --model gpt2 --lag 1
```

All scripts use a fixed random seed (42).

## Citation

```bibtex
@misc{muletvazquez2026marginadaptive,
  title  = {Margin-Adaptive Retrieval Complexity: A Separation Between Normalized and Max-Plus Attention},
  author = {Mulet V\'azquez, Alfonso},
  year   = {2026},
  note   = {arXiv preprint; identifier forthcoming}
}
```

## License

MIT License — see [LICENSE](LICENSE).
