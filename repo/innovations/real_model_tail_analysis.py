"""INNOVATION / PHASE 3 (v3) -- per-HEAD KS-test of the Section 7 tail claim on GPT-2 small.

WHICH CLAIM
    Section 7 asserts "training pushes attention logits into the exponential tail" -- the paper's
    only unvalidated empirical claim, and the highest-value thing this run can produce. The
    average-case theory (Thm 7.2) is practically relevant only if real score top-gaps resemble the
    exponential (heavy) regime more than Gaussian or (worst case) bounded/uniform.

WHAT IT DOES
    Loads GPT-2 small (CPU), runs a fixed 512-token prompt, and for EVERY (layer, head) extracts the
    per-query attention top-gap  Delta = s_(1) - s_(2)  of the pre-softmax logits. For each head it
    fits Exponential, Gaussian (half-normal, since gaps are >=0), and Uniform by maximum likelihood
    and scores the fit by the Kolmogorov-Smirnov statistic (lower = better). It reports:
      - per-head best-fit family,
      - a per-LAYER summary table (counts of heads per regime + mean KS gaps),
      - whether exponential dominates overall, and whether it STRENGTHENS with depth (later layers).

PASS / VERDICT
    SUPPORTED if (a) exponential is the plurality best-fit across heads AND (b) uniform/bounded is
    rare (<= a small fraction). We also report the depth trend honestly; the test CAN refute the
    "strengthens with depth" sub-claim (and, in earlier per-layer analysis, the trend was the
    OPPOSITE -- early layers heavier-tailed -- which we report rather than hide).

CPU-only, seed 42. Run: python innovations/real_model_tail_analysis.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np

SEED = 42
np.random.seed(SEED)
FAMILIES = ("exponential", "gaussian", "uniform")


def ks_fits(x: np.ndarray) -> dict:
    """KS statistic of x (>=0 top-gaps) against ML-fit Exp, half-Gaussian, Uniform. Lower=better."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    out = {}
    # Exponential: loc at min, scale = mean of shifted
    loc = x.min() - 1e-12
    out["exponential"] = stats.kstest(x, "expon", args=(loc, max((x - loc).mean(), 1e-12))).statistic
    # Half-normal (gaps are nonnegative): fit scale by second moment about 0
    sigma = max(np.sqrt(np.mean(x ** 2)), 1e-12)
    out["gaussian"] = stats.kstest(x, "halfnorm", args=(0.0, sigma)).statistic
    # Uniform on [min,max]
    out["uniform"] = stats.kstest(x, "uniform", args=(x.min(), max(x.max() - x.min(), 1e-12))).statistic
    return out


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except Exception as e:
        print(f"INCONCLUSIVE: deps unavailable ({type(e).__name__}: {e}).")
        return

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    base = ("In the study of attention, the relevance score of each key reflects how strongly it "
            "matches the current query; whether these scores concentrate on one token or spread "
            "across many shapes both the model's expressivity and its computational cost. ")
    ids = tok(base * 24, return_tensors="pt", truncation=True, max_length=512)
    T = ids["input_ids"].shape[1]
    nhead, d = model.config.n_head, model.config.n_embd
    dh = d // nhead
    L = len(model.transformer.h)
    print(f"GPT-2 small | T={T} tokens | {L} layers x {nhead} heads | per-HEAD KS tail fit\n")

    per_layer = []   # list of dicts: counts per family + mean exp-KS for the layer
    head_best = []   # (layer, head, best_family)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states
        for li, blk in enumerate(model.transformer.h):
            x = blk.ln_1(hs[li])
            q, k, _v = blk.attn.c_attn(x).split(d, dim=2)
            q = q.view(1, T, nhead, dh).permute(0, 2, 1, 3)
            k = k.view(1, T, nhead, dh).permute(0, 2, 1, 3)
            logits = ((q @ k.transpose(-1, -2)) / np.sqrt(dh))[0].numpy()  # [head, Tq, Tk]
            counts = {f: 0 for f in FAMILIES}
            exp_ks = []
            for hh in range(nhead):
                gaps = []
                for qp in range(4, T):
                    row = np.sort(logits[hh, qp, : qp + 1])
                    gaps.append(row[-1] - row[-2])
                gaps = np.array(gaps)
                gaps = gaps[gaps > 0]
                if gaps.size < 20:
                    continue
                ks = ks_fits(gaps)
                best = min(ks, key=ks.get)
                counts[best] += 1
                exp_ks.append(ks["exponential"])
                head_best.append((li, hh, best))
            per_layer.append((li, counts, float(np.mean(exp_ks)) if exp_ks else float("nan")))

    # per-layer summary table
    print(f"{'layer':>5} | {'exp':>4} {'gauss':>5} {'unif':>4} | {'mean KS(exp)':>12} | plurality")
    for li, counts, mexp in per_layer:
        plur = max(counts, key=counts.get)
        print(f"{li:>5} | {counts['exponential']:>4} {counts['gaussian']:>5} {counts['uniform']:>4} "
              f"| {mexp:>12.3f} | {plur}")

    # aggregate verdict
    from collections import Counter
    allbest = Counter(b for _, _, b in head_best)
    total = sum(allbest.values())
    print(f"\n  per-HEAD best-fit tally over {total} heads: {dict(allbest)}")
    exp_frac = allbest["exponential"] / total
    unif_frac = allbest["uniform"] / total
    # depth trend: compare exp-share in first vs second half of layers
    half = len(per_layer) // 2
    early_exp = sum(c["exponential"] for _, c, _ in per_layer[:half])
    late_exp = sum(c["exponential"] for _, c, _ in per_layer[half:])
    early_tot = sum(sum(c.values()) for _, c, _ in per_layer[:half])
    late_tot = sum(sum(c.values()) for _, c, _ in per_layer[half:])
    early_share = early_exp / max(early_tot, 1)
    late_share = late_exp / max(late_tot, 1)
    print(f"  exponential best-fit fraction: {exp_frac:.2f}   uniform/bounded fraction: {unif_frac:.2f}")
    print(f"  exp share early layers={early_share:.2f}  vs late layers={late_share:.2f}  "
          f"(strengthens with depth: {late_share > early_share})")
    supported = exp_frac >= 0.4 and unif_frac <= 0.15
    print(f"\n  Section 7 claim 'logits favor the exponential/heavy tail': "
          f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    if late_share <= early_share:
        print("  NOTE (honest): the 'strengthens with DEPTH' sub-claim is NOT supported -- the "
              "heavy-tail share is at least as high in EARLY layers. Report this; do not overclaim.")


if __name__ == "__main__":
    main()
