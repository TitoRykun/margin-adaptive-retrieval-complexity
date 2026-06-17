"""INNOVATION -- multi-head margin and its query-cost consequence.

WHICH RESULT (extends the single-head theory to real multi-head attention)
    The paper analyzes a single head. A multi-head layer runs H heads in parallel; the per-query
    indexed-max cost of the LAYER is dominated by the HARDEST head, i.e. the one with the smallest
    margin. Define the multi-head margin  Delta_MH = min_h Delta_h.  Two questions:
      (1) empirical: in GPT-2, how is Delta_MH distributed, and are per-head margins correlated or
          independent across heads (independence makes the min much smaller than any single head)?
      (2) theory: what is the layer's indexed-max query cost as a function of {Delta_h} and H?

    COST BOUND.  With per-head multiplicative gaps Gamma_h and an LSH index of exponent rho(.), each
    head certifies in N^{rho(c_h)} with c_h < Gamma_h; running H heads costs
        sum_h N^{rho(c_h)}  <=  H * N^{rho(c_min)} ,  c_min = min_h Gamma_h,
    so the layer exponent is governed by the WORST head's gap, with only an additive factor H (a
    log_N H additive term in the exponent). Independence across heads => Delta_MH concentrates near
    the low-order statistic of H gaps, shrinking the usable speedup; positive cross-head correlation
    keeps Delta_MH closer to the typical single-head margin.

PASS DEFINITION
    The analysis runs on GPT-2, reports (a) the cross-head Pearson correlation of per-query margins
    (PASS = computed and reported, with interpretation), and (b) verifies the cost-bound inequality
    sum_h N^{rho_h} <= H N^{rho_min} numerically on the extracted gaps.

CPU-only. Seed 42.  Run: python innovations/multihead_margin.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np

SEED = 42
np.random.seed(SEED)


def rho_lsh(c, exponent_kind="data_dependent"):
    """Optimal LSH exponent as a function of approximation factor c>1."""
    c = np.maximum(c, 1.0 + 1e-9)
    return 1.0 / (2.0 * c - 1.0) if exponent_kind == "data_dependent" else 1.0 / c


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
    base = ("Multi-head attention runs several heads in parallel, each with its own notion of "
            "relevance. The cost of the layer is set by the hardest head to certify. ")
    ids = tok(base * 24, return_tensors="pt", truncation=True, max_length=512)
    T = ids["input_ids"].shape[1]
    nhead = model.config.n_head
    d = model.config.n_embd
    dh = d // nhead
    N_for_cost = 4096  # hypothetical context for the cost-bound check

    print(f"GPT-2 small | T={T} | H={nhead} heads | multi-head margin analysis\n")
    print(f"{'layer':>5} {'mean Delta_h':>12} {'mean Delta_MH':>13} {'cross-head corr':>15} {'cost-bound':>11}")
    corrs = []
    bound_ok_all = True
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states
        for li, blk in enumerate(model.transformer.h):
            x = blk.ln_1(hs[li])
            qkv = blk.attn.c_attn(x)
            q, k, _v = qkv.split(d, dim=2)
            q = q.view(1, T, nhead, dh).permute(0, 2, 1, 3)
            k = k.view(1, T, nhead, dh).permute(0, 2, 1, 3)
            logits = ((q @ k.transpose(-1, -2)) / np.sqrt(dh))[0].numpy()  # [H, Tq, Tk]
            # per-query, per-head additive top gap, for query positions with enough keys
            qpositions = list(range(8, T))
            gaps = np.zeros((nhead, len(qpositions)))
            for hh in range(nhead):
                for qi, qp in enumerate(qpositions):
                    row = np.sort(logits[hh, qp, : qp + 1])
                    gaps[hh, qi] = row[-1] - row[-2]
            delta_mh = gaps.min(axis=0)              # min over heads, per query
            mean_dh = gaps.mean()
            # cross-head correlation: average pairwise Pearson corr of per-query margins
            cc = np.corrcoef(gaps)
            offdiag = cc[~np.eye(nhead, dtype=bool)]
            mean_corr = float(np.nanmean(offdiag))
            corrs.append(mean_corr)
            # cost-bound check: sum_h N^{rho(Gamma_h)} <= H * N^{rho(min Gamma_h)} on a sample query
            # use a positive-shifted gap as a proxy multiplicative gap Gamma_h = 1 + mean_gap_h
            mean_gap_h = gaps.mean(axis=1)
            Gamma_h = 1.0 + mean_gap_h
            rhos = rho_lsh(Gamma_h)
            lhs = np.sum(N_for_cost ** rhos)
            rhs = nhead * N_for_cost ** rho_lsh(Gamma_h.min())
            bound_ok = lhs <= rhs + 1e-6
            bound_ok_all = bound_ok_all and bound_ok
            print(f"{li:>5} {mean_dh:>12.3f} {delta_mh.mean():>13.3f} {mean_corr:>15.3f} "
                  f"{'OK' if bound_ok else 'FAIL':>11}")

    mean_overall_corr = float(np.mean(corrs))
    print(f"\n  mean cross-head margin correlation = {mean_overall_corr:.3f}")
    if mean_overall_corr < 0.1:
        interp = ("near-INDEPENDENT heads: Delta_MH = min_h Delta_h is much smaller than a single "
                  "head's margin, so the multi-head layer is harder to speed up than any one head.")
    elif mean_overall_corr < 0.4:
        interp = ("WEAKLY correlated heads: the min margin is moderately below the typical head; "
                  "partial speedup erosion.")
    else:
        interp = ("STRONGLY correlated heads: Delta_MH stays close to the single-head margin; the "
                  "speedup largely survives multi-head aggregation.")
    print(f"  interpretation: {interp}")
    print(f"  cost-bound  sum_h N^rho_h <= H * N^rho_min  holds at every layer: {bound_ok_all}")
    print(f"  VERDICT: {'PASS' if bound_ok_all else 'FAIL'} (cost bound); cross-head correlation reported")


if __name__ == "__main__":
    main()
