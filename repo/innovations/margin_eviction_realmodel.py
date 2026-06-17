"""INNOVATION 1b -- margin-aware KV-cache eviction on REAL model attention.

WHY THIS, NOT THE SYNTHETIC VERSION
    margin_eviction.py showed a margin-criticality policy beats LRU/random on a SYNTHETIC stream of
    Gaussian keys. A skeptic rightly discounts that. This script runs the same idea on the ACTUAL
    attention of pretrained models (GPT-2, Pythia-160M), against the baselines a real KV-cache
    system would use, with the metric that actually governs generation quality.

THE HONEST METRIC
    KV-cache eviction does not "retrieve a key" -- it changes the attention OUTPUT. For a decoder,
    each query position q attends (causally) over keys 0..q with weights softmax(scores) and mixes
    their value vectors. The FULL output o_full(q) = sum_j softmax(s)_j v_j over all causal j. Under
    a cache budget B, a policy keeps only B of those keys; the cached output renormalises softmax
    over the kept set. We measure the RELATIVE L2 ERROR  ||o_cache - o_full|| / ||o_full||  averaged
    over query positions, heads, prompts -- lower is better. This is exactly what degrades when a
    cache is too small, so a policy that minimises it is the useful one.

POLICIES (eviction = which keys to KEEP within budget B, decided per query from info available
    up to that query -- causal, no peeking at the future)
    - recency : keep the most recent B keys (the standard, very strong causal-LM baseline)
    - random  : keep B random kept-eligible keys
    - attn    : heavy-hitter / H2O-style -- keep the B keys with largest accumulated attention mass
                so far (a strong, widely used real baseline)
    - margin  : THEORY POLICY -- keep the B keys with highest running "criticality" = how often the
                key has been at/near the decision boundary (within the top-gap) of recent queries;
                evict keys that are robustly far from ever deciding (the paper's margin intuition).

HONEST EXPECTATION
    Recency is a famously strong causal baseline; beating it is hard and we do NOT assume margin
    will. The point is a faithful measurement: report where margin lands relative to recency, attn,
    random -- a win, a tie, or a measured loss, all are reported, none hidden.

MEASURED RESULT: at --lag 1 (re-decide the kept set every query) margin
    has ~half the output error of recency/attn/random on GPT-2 and Pythia. BUT this advantage does
    NOT survive lag: with --lag 5 or 20 (decide once, then serve several future queries before
    re-deciding -- the realistic cache regime) margin's edge over RECENCY collapses and recency
    wins, because adjacent queries attend to nearly the same keys and lag-1 criticality is mostly
    short-range autocorrelation with recency. Margin DOES still beat attn/H2O at all lags. So the
    honest verdict is: not a deployable improvement over the recency baseline; a modest, robust
    improvement only over heavy-hitter. Use --lag to reproduce the collapse.

CPU-only, seed 42.
    python innovations/margin_eviction_realmodel.py --model gpt2 --lag 1
    python innovations/margin_eviction_realmodel.py --model gpt2 --lag 20   # advantage gone
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np

from multimodel_tail_analysis import DEFAULT_PROMPT  # reuse the same prompt source

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "out"

PROMPTS = {
    "technical": DEFAULT_PROMPT,
    "narrative": ("She walked to the harbour before dawn, when the boats were still dark shapes "
                  "against a paler sky, and waited for the captain who had promised to take her "
                  "north past the last island anyone had bothered to name. "),
    "code": ("def merge(left, right):\n    out = []\n    i = j = 0\n    while i < len(left) and j "
             "< len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i "
             "+= 1\n        else:\n            out.append(right[j]); j += 1\n    return out\n"),
}


def capture_scores_and_values(model, tok, text: str, repeat: int, max_len: int):
    """Return (per-layer pre-softmax scores [head,Tq,Tk], per-layer value tensors [head,Tk,dh], T).

    Scores: wrap F.softmax (architecture-agnostic, already has scale/RoPE/mask). Values: register a
    forward hook on each block's value projection. GPT-2 packs QKV in c_attn (v = last third);
    GPT-NeoX packs per-head in query_key_value (v = last of 3 per head).
    """
    import torch
    import torch.nn.functional as F

    ids = tok(text * repeat, return_tensors="pt", truncation=True, max_length=max_len)
    t = ids["input_ids"].shape[1]
    cfg = model.config
    nhead = getattr(cfg, "n_head", None) or getattr(cfg, "num_attention_heads")
    d = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size")
    dh = d // nhead

    scores: list[np.ndarray] = []
    orig = F.softmax

    def spy(inp, *a, **k):  # noqa: A002
        if inp.dim() == 4 and inp.shape[1] == nhead and inp.shape[-1] == t:
            scores.append(inp.detach()[0].float().numpy())
        return orig(inp, *a, **k)

    values: dict[int, np.ndarray] = {}
    handles = []

    def make_v_hook(li: int, family: str):
        def hook(module, inp, out):
            packed = out[0] if isinstance(out, tuple) else out
            packed = packed[0]  # [T, *]
            if family == "gpt2":                       # [T, 3d] -> v is last d
                v = packed[:, 2 * d:]
                v = v.view(t, nhead, dh).permute(1, 0, 2)         # [head, T, dh]
            else:                                       # neox [T, 3*d] interleaved per head
                v = packed.view(t, nhead, 3 * dh)[:, :, 2 * dh:]  # [T, head, dh]
                v = v.permute(1, 0, 2)                            # [head, T, dh]
            values[li] = v.detach().float().numpy()
        return hook

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):     # GPT-2
        for li, blk in enumerate(model.transformer.h):
            handles.append(blk.attn.c_attn.register_forward_hook(make_v_hook(li, "gpt2")))
    elif hasattr(model, "gpt_neox"):                                          # Pythia/NeoX
        for li, blk in enumerate(model.gpt_neox.layers):
            handles.append(blk.attention.query_key_value.register_forward_hook(make_v_hook(li, "neox")))
    else:
        raise RuntimeError(f"value capture unsupported for {type(model).__name__}")

    F.softmax = spy
    try:
        with torch.no_grad():
            model(**ids)
    finally:
        F.softmax = orig
        for h in handles:
            h.remove()

    L = min(len(scores), len(values))
    return [scores[i] for i in range(L)], [values[i] for i in range(L)], t


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def output_rel_error(score_row: np.ndarray, vals: np.ndarray, keep_idx: np.ndarray) -> float:
    """rel L2 error of cached output (softmax over keep_idx) vs full causal output. score_row:[K],
    vals:[K,dh] for the causal prefix; keep_idx subset of 0..K-1 (must be nonempty)."""
    w_full = _softmax(score_row)
    o_full = w_full @ vals
    w_c = _softmax(score_row[keep_idx])
    o_c = w_c @ vals[keep_idx]
    denom = np.linalg.norm(o_full) + 1e-12
    return float(np.linalg.norm(o_c - o_full) / denom)


def eval_head(scores_head: np.ndarray, vals_head: np.ndarray, budget_frac: float,
              start: int, lag: int = 1) -> dict:
    """Mean output rel-error per policy over query positions q >= start, head fixed.

    For each query q, the causal prefix is keys 0..q. Budget B = max(1, round(budget_frac*(q+1))).
    Criticality / attention-mass are accumulated CAUSALLY. `lag` controls how stale the keep-decision
    is: the kept set for query q is chosen from the criticality/attn snapshot as of query q-lag.
    lag=1 means re-decide every step (best case for adaptive policies); lag>1 models a cache that is
    re-scored only periodically -- the realistic regime, and the one that exposes whether an apparent
    advantage was just short-range autocorrelation with recency.
    """
    T = scores_head.shape[0]
    crit = np.zeros(T)        # margin criticality, EMA of proximity-to-winner
    attn_mass = np.zeros(T)   # accumulated attention weight (heavy-hitter)
    crit_snap = [None] * T    # snapshots for lagged decisions
    attn_snap = [None] * T
    ema = 0.2
    rng = np.random.default_rng(SEED)
    errs = {p: [] for p in ("recency", "random", "attn", "margin")}

    for q in range(T):
        K = q + 1
        srow = scores_head[q, :K]
        vrow = vals_head[:K]
        B = max(1, int(round(budget_frac * K)))
        si = q - lag
        crit_src = crit_snap[si] if si >= 0 and crit_snap[si] is not None else np.zeros(T)
        attn_src = attn_snap[si] if si >= 0 and attn_snap[si] is not None else np.zeros(T)
        if B < K:  # only meaningful when we actually evict
            keep_rec = np.arange(K - B, K)                       # recency: last B keys
            keep_rnd = rng.choice(K, size=B, replace=False)      # random
            keep_attn = np.argsort(attn_src[:K])[-B:]            # heavy-hitter (lagged)
            keep_mar = np.argsort(crit_src[:K])[-B:]             # margin criticality (lagged)
            for name, keep in (("recency", keep_rec), ("random", keep_rnd),
                               ("attn", keep_attn), ("margin", keep_mar)):
                if q >= start:
                    errs[name].append(output_rel_error(srow, vrow, keep))
        # ---- update causal stats using THIS query's weights (available after q) ----
        w = _softmax(srow)
        attn_mass[:K] += w
        gap_to_top = srow.max() - srow
        proximity = np.exp(-gap_to_top)               # ~1 near winner
        crit[:K] = (1 - ema) * crit[:K] + ema * proximity
        crit_snap[q] = crit.copy()
        attn_snap[q] = attn_mass.copy()
    return {p: (float(np.mean(v)) if v else float("nan")) for p, v in errs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--repeat", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--max-layers", type=int, default=6, help="cap layers for CPU speed")
    ap.add_argument("--lag", type=int, default=1,
                    help="staleness of the keep-decision (1=re-decide each step; >1=realistic).")
    args = ap.parse_args()

    np.random.seed(SEED)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # noqa: BLE001
        print(f"INCONCLUSIVE: deps unavailable ({type(e).__name__}: {e}).")
        return
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation="eager", dtype=torch.float32).eval()
    except Exception as e:  # noqa: BLE001
        print(f"INCONCLUSIVE: could not load {args.model} ({type(e).__name__}: {e}).")
        return

    print(f"Real-model margin eviction | {args.model} | lag={args.lag} | metric = output rel-L2 "
          f"error vs full cache (LOWER better)\n")
    # budget -> policy -> list of per-(prompt,layer,head) mean errors
    agg = {b: {p: [] for p in ("recency", "random", "attn", "margin")} for b in args.budgets}
    for pname, ptext in PROMPTS.items():
        scores, values, T = capture_scores_and_values(model, tok, ptext, args.repeat, args.max_len)
        L = min(len(scores), args.max_layers)
        nhead = scores[0].shape[0]
        start = T // 4  # measure on the second-half-ish, where the cache actually bites
        for b in args.budgets:
            for li in range(L):
                for hh in range(nhead):
                    res = eval_head(scores[li][hh], values[li][hh], b, start, lag=args.lag)
                    for p in agg[b]:
                        if np.isfinite(res[p]):
                            agg[b][p].append(res[p])

    print(f"{'budget':>7} | {'recency':>8} {'attn(H2O)':>9} {'margin':>8} {'random':>8} | best")
    summary = {}
    for b in args.budgets:
        means = {p: float(np.mean(agg[b][p])) for p in agg[b]}
        best = min(means, key=means.get)
        print(f"{b:>7.2f} | {means['recency']:>8.4f} {means['attn']:>9.4f} {means['margin']:>8.4f} "
              f"{means['random']:>8.4f} | {best}")
        summary[str(b)] = {"means": means, "best": best}

    # honest verdict
    print()
    margin_beats_random = all(summary[str(b)]["means"]["margin"] <=
                              summary[str(b)]["means"]["random"] + 1e-9 for b in args.budgets)
    margin_vs_attn = [summary[str(b)]["means"]["margin"] <= summary[str(b)]["means"]["attn"] + 1e-9
                      for b in args.budgets]
    margin_vs_rec = [summary[str(b)]["means"]["margin"] <= summary[str(b)]["means"]["recency"] + 1e-9
                     for b in args.budgets]
    print(f"  margin <= random at all budgets : {margin_beats_random}")
    print(f"  margin <= attn(H2O) per budget  : {margin_vs_attn}")
    print(f"  margin <= recency per budget    : {margin_vs_rec}")
    print("  (recency is a famously strong causal baseline; not beating it is the expected, honest "
          "outcome. The useful question is whether margin is competitive with attn/H2O.)")

    OUT_DIR.mkdir(exist_ok=True)
    safe = args.model.replace("/", "_")
    path = OUT_DIR / f"eviction_realmodel_{safe}.json"
    with open(path, "w") as fh:
        json.dump({"model": args.model, "seed": SEED, "lag": args.lag, "budgets": args.budgets,
                   "metric": "output_rel_L2_error_vs_full", "summary": summary}, fh, indent=2)
    print(f"\n  raw results saved: {path}")


if __name__ == "__main__":
    main()
