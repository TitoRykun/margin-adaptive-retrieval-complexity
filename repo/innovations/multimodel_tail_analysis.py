"""Multi-model per-HEAD KS-test of the Section 7 tail claim (model-agnostic).

WHY
    Section 7's empirical claim ("real attention logit top-gaps favour the exponential/heavy
    regime") was only ever checked on GPT-2 small. A single-architecture result cannot support a
    general statement about "trained transformers." This script reruns the IDENTICAL per-head KS
    methodology on an arbitrary HF causal-LM so the GPT-2 finding can be compared against a second
    family (Pythia-160M, a GPT-NeoX-style model) on equal footing.

WHAT IT DOES (identical methodology to real_model_tail_analysis.py)
    For EVERY (layer, head) it extracts the per-query attention top-gap  Delta = s_(1) - s_(2)  of
    the PRE-softmax logits, fits Exponential / half-Gaussian / Uniform by the same closed-form ML
    estimates, and scores each by the Kolmogorov-Smirnov statistic (lower = better). It reports the
    per-head best-fit plurality and the early-vs-late-layer depth trend.

HOW IT IS MODEL-AGNOSTIC
    Reconstructing Q@K by hand is architecture-specific and error-prone (GPT-2 concatenates QKV as
    three contiguous blocks; GPT-NeoX interleaves them per-head and then applies ROTARY embeddings
    to Q,K before the score product). To avoid that entire class of bugs we capture the model's
    ACTUAL pre-softmax scores: during the forward pass we temporarily wrap the attention softmax and
    record the exact tensor the model feeds into it. That tensor already has RoPE/ALiBi, the 1/sqrt(d)
    scale, and the causal mask applied by the model itself -- so it is the true attention logit
    matrix for whatever architecture is loaded. The KS math is shared verbatim with the GPT-2 script
    (see `ks_fits`), so the ONLY thing differing between runs is the model.

    NOTE on masking: the captured scores contain large negative fill (e.g. -inf or a big negative
    constant) at the masked-out future positions. We restrict each query's gap to the unmasked
    causal prefix [0..qp] BEFORE computing the top-gap, so the mask fill never enters the statistic.

CPU-only, seed 42.
    python innovations/multimodel_tail_analysis.py --model EleutherAI/pythia-160m
    python innovations/multimodel_tail_analysis.py --model gpt2          # sanity: must match v3
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 42
FAMILIES = ("exponential", "gaussian", "uniform")
OUT_DIR = Path(__file__).resolve().parent / "out"


def ks_fits(x: np.ndarray) -> dict:
    """KS statistic of x (>=0 top-gaps) vs ML-fit Exp, half-Gaussian, Uniform. Lower=better.

    Byte-for-byte the same estimator as real_model_tail_analysis.py so cross-model results are
    directly comparable (no methodology drift).
    """
    from scipy import stats

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    out = {}
    loc = x.min() - 1e-12
    out["exponential"] = stats.kstest(x, "expon", args=(loc, max((x - loc).mean(), 1e-12))).statistic
    sigma = max(np.sqrt(np.mean(x ** 2)), 1e-12)
    out["gaussian"] = stats.kstest(x, "halfnorm", args=(0.0, sigma)).statistic
    out["uniform"] = stats.kstest(x, "uniform", args=(x.min(), max(x.max() - x.min(), 1e-12))).statistic
    return out


def gaps_from_logits(logits_head: np.ndarray, t: int) -> np.ndarray:
    """Per-query causal top-gap s_(1)-s_(2) for one head's [Tq,Tk] logit matrix. >0 only."""
    gaps = []
    for qp in range(4, t):
        row = np.sort(logits_head[qp, : qp + 1])
        gaps.append(row[-1] - row[-2])
    g = np.asarray(gaps)
    return g[g > 0]


DEFAULT_PROMPT = (
    "In the study of attention, the relevance score of each key reflects how strongly it "
    "matches the current query; whether these scores concentrate on one token or spread "
    "across many shapes both the model's expressivity and its computational cost. ")


def collect_logits(model, tok, prompt_repeat: int, max_len: int, text: str | None = None):
    """Returns list over layers of [head, Tq, Tk] PRE-softmax logit arrays, plus (L, nhead, T).

    Architecture-agnostic: captures the exact tensor the model feeds into its attention softmax by
    wrapping torch.nn.functional.softmax for the duration of one forward pass. Each captured 4-D
    score tensor [1, head, Tq, Tk] is recorded in call order, which for a standard decoder is one
    per layer, in depth order. This sidesteps all QKV-layout and RoPE reconstruction.

    `text` defaults to DEFAULT_PROMPT (the original single-prompt run, repeated `prompt_repeat`
    times). Passing an explicit `text` lets the robustness harness sweep distinct prompts; the
    string is repeated until truncated to `max_len`, so every prompt fills the same context length.
    """
    import torch
    import torch.nn.functional as F

    base = text if text is not None else DEFAULT_PROMPT
    ids = tok(base * prompt_repeat, return_tensors="pt", truncation=True, max_length=max_len)
    t = ids["input_ids"].shape[1]

    cfg = model.config
    nhead = getattr(cfg, "n_head", None) or getattr(cfg, "num_attention_heads")

    captured: list[np.ndarray] = []
    orig_softmax = F.softmax

    def spy_softmax(input, *args, **kwargs):  # noqa: A002 (match torch signature)
        # The attention scores are the only 4-D [B, head, Tq, Tk] softmax in a decoder block.
        if input.dim() == 4 and input.shape[1] == nhead and input.shape[-1] == t:
            captured.append(input.detach()[0].float().numpy())  # [head, Tq, Tk]
        return orig_softmax(input, *args, **kwargs)

    F.softmax = spy_softmax
    try:
        with torch.no_grad():
            model(**ids)
    finally:
        F.softmax = orig_softmax

    if not captured:
        raise RuntimeError("captured no attention-score tensors; model may not call F.softmax "
                           "(e.g. fused/SDPA path) -- load with attn_implementation='eager'.")
    L = len(captured)
    return captured, (L, nhead, t)


def analyse(per_layer_logits, nhead: int) -> dict:
    """Per-head KS best-fit tally + per-layer table + depth trend. Returns a JSON-able dict."""
    per_layer = []
    head_best = []
    for li, logits in enumerate(per_layer_logits):
        t = logits.shape[1]
        counts = {f: 0 for f in FAMILIES}
        exp_ks = []
        for hh in range(nhead):
            gaps = gaps_from_logits(logits[hh], t)
            if gaps.size < 20:
                continue
            ks = ks_fits(gaps)
            best = min(ks, key=ks.get)
            counts[best] += 1
            exp_ks.append(ks["exponential"])
            head_best.append({"layer": li, "head": hh, "best": best,
                              "ks": {k: round(v, 4) for k, v in ks.items()}})
        per_layer.append({"layer": li, "counts": counts,
                          "mean_ks_exp": float(np.mean(exp_ks)) if exp_ks else float("nan")})

    allbest = Counter(h["best"] for h in head_best)
    total = max(sum(allbest.values()), 1)
    half = len(per_layer) // 2
    early_exp = sum(p["counts"]["exponential"] for p in per_layer[:half])
    late_exp = sum(p["counts"]["exponential"] for p in per_layer[half:])
    early_tot = max(sum(sum(p["counts"].values()) for p in per_layer[:half]), 1)
    late_tot = max(sum(sum(p["counts"].values()) for p in per_layer[half:]), 1)
    return {
        "per_layer": per_layer,
        "head_best": head_best,
        "tally": dict(allbest),
        "exp_frac": allbest["exponential"] / total,
        "gauss_frac": allbest["gaussian"] / total,
        "unif_frac": allbest["uniform"] / total,
        "early_exp_share": early_exp / early_tot,
        "late_exp_share": late_exp / late_tot,
        "n_heads_scored": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--repeat", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    np.random.seed(SEED)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import torch  # noqa: F401
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        print(f"INCONCLUSIVE: deps unavailable ({type(e).__name__}: {e}).")
        return

    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        # Pin float32 explicitly: some checkpoints otherwise load a dtype that yields NaN attention
        # scores on long contexts via the eager path (verified on Pythia-160M). float32 is finite.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation="eager", dtype=torch.float32).eval()
    except Exception as e:
        print(f"INCONCLUSIVE: could not load {args.model} ({type(e).__name__}: {e}).")
        return

    per_layer_logits, (L, nhead, t) = collect_logits(model, tok, args.repeat, args.max_len)
    print(f"{args.model} | T={t} tokens | {L} layers x {nhead} heads | per-HEAD KS tail fit\n")

    res = analyse(per_layer_logits, nhead)

    print(f"{'layer':>5} | {'exp':>4} {'gauss':>5} {'unif':>4} | {'mean KS(exp)':>12} | plurality")
    for p in res["per_layer"]:
        c = p["counts"]
        plur = max(c, key=c.get)
        print(f"{p['layer']:>5} | {c['exponential']:>4} {c['gaussian']:>5} {c['uniform']:>4} "
              f"| {p['mean_ks_exp']:>12.3f} | {plur}")

    print(f"\n  per-HEAD best-fit tally over {res['n_heads_scored']} heads: {res['tally']}")
    print(f"  exponential fraction: {res['exp_frac']:.2f}   gaussian: {res['gauss_frac']:.2f}   "
          f"uniform/bounded: {res['unif_frac']:.2f}")
    print(f"  exp share early={res['early_exp_share']:.2f} vs late={res['late_exp_share']:.2f}  "
          f"(strengthens with depth: {res['late_exp_share'] > res['early_exp_share']})")
    supported = res["exp_frac"] >= 0.4 and res["unif_frac"] <= 0.15
    print(f"\n  Section 7 'logits favour exponential/heavy tail': "
          f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    if res["late_exp_share"] <= res["early_exp_share"]:
        print("  NOTE (honest): 'strengthens with DEPTH' NOT supported -- exp share is at least as "
              "high in EARLY layers.")

    OUT_DIR.mkdir(exist_ok=True)
    safe = args.model.replace("/", "_")
    path = OUT_DIR / f"tail_ks_{safe}.json"
    with open(path, "w") as fh:
        json.dump({"model": args.model, "T": t, "L": L, "nhead": nhead,
                   "seed": SEED, **res}, fh, indent=2)
    print(f"\n  raw per-head results saved: {path}")


if __name__ == "__main__":
    main()
