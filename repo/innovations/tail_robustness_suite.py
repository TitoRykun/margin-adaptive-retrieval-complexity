"""Cross-prompt / cross-architecture robustness of the Section 7 attention-tail finding.

WHY
    The single-prompt per-head KS check showed GPT-2 (plurality bounded) and Pythia (plurality
    exponential) disagree on the dominant attention top-gap regime. A reviewer's first objection is:
    is that an architecture effect, or just one prompt? This suite reruns the IDENTICAL methodology
    (imported verbatim from multimodel_tail_analysis: ks_fits, gaps_from_logits, collect_logits,
    analyse) over several DISTINCT prompts and several architectures, and reports:
      - per (model, prompt) regime tally,
      - per-model pooled tally and the cross-prompt spread (min/max exponential share),
      - whether the per-prompt PLURALITY regime is stable within each model,
      - the early-vs-late depth trend per model (the one sub-claim that was consistent before).

WHAT IS CONTROLLED
    Same KS estimator, same gap definition, same context length, same seed -> the only things that
    vary are (model, prompt). That isolates architecture and prompt effects from methodology.

CPU-only, seed 42.
    python innovations/tail_robustness_suite.py
    python innovations/tail_robustness_suite.py --models gpt2 EleutherAI/pythia-160m facebook/opt-125m
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np

from multimodel_tail_analysis import analyse, collect_logits  # shared, identical methodology

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "out"

# Five genuinely distinct prompts: technical prose, narrative, source code, dialogue, structured list.
# Each is repeated to fill the context window, so content (not length) is what differs.
PROMPTS: dict[str, str] = {
    "technical": (
        "The transformer computes, for every query, a distribution over keys via the softmax of "
        "scaled dot products; the entropy of that distribution governs how many tokens materially "
        "contribute to the output, and therefore how sharply the layer can retrieve. "),
    "narrative": (
        "She walked to the harbour before dawn, when the boats were still dark shapes against a "
        "paler sky, and waited for the one captain who had promised, years ago, to take her north "
        "past the last island anyone had bothered to name. "),
    "code": (
        "def merge(left, right):\n    out = []\n    i = j = 0\n    while i < len(left) and j < "
        "len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n   "
        "     else:\n            out.append(right[j]); j += 1\n    return out + left[i:] + right[j:]\n"),
    "dialogue": (
        "\"Did you read the report?\" she asked. \"Most of it,\" he said. \"And?\" \"And I think the "
        "numbers don't add up the way they want them to.\" \"You always think that.\" \"Because it "
        "keeps being true.\" "),
    "list": (
        "Groceries to buy: ripe tomatoes, two loaves of sourdough, a dozen eggs, olive oil, fresh "
        "basil, parmesan, black pepper, sea salt, a bottle of red wine, dark chocolate, coffee "
        "beans, and oranges for the morning. "),
}


def run_model(model_name: str, repeat: int, max_len: int) -> dict | None:
    """Load a model once, run all prompts, return per-prompt + pooled stats. None on load failure."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="eager", dtype=torch.float32).eval()
    except Exception as e:  # noqa: BLE001 -- report, don't crash the whole suite
        print(f"  [{model_name}] INCONCLUSIVE: could not load ({type(e).__name__}: {e}).")
        return None

    cfg = model.config
    nhead = getattr(cfg, "n_head", None) or getattr(cfg, "num_attention_heads")

    per_prompt = {}
    for pname, ptext in PROMPTS.items():
        try:
            logits, (L, _nh, t) = collect_logits(model, tok, repeat, max_len, text=ptext)
        except Exception as e:  # noqa: BLE001
            print(f"  [{model_name}/{pname}] skipped ({type(e).__name__}: {e}).")
            continue
        res = analyse(logits, nhead)
        if not np.isfinite(res["exp_frac"]):
            print(f"  [{model_name}/{pname}] skipped (non-finite stats; possible NaN logits).")
            continue
        per_prompt[pname] = {
            "T": t, "L": L, "tally": res["tally"],
            "exp_frac": res["exp_frac"], "gauss_frac": res["gauss_frac"],
            "unif_frac": res["unif_frac"],
            "plurality": max(res["tally"], key=res["tally"].get),
            "early_exp_share": res["early_exp_share"],
            "late_exp_share": res["late_exp_share"],
        }

    if not per_prompt:
        return {"model": model_name, "nhead": nhead, "L": cfg.num_hidden_layers
                if hasattr(cfg, "num_hidden_layers") else None, "per_prompt": {}, "pooled": None}

    exp_fracs = [p["exp_frac"] for p in per_prompt.values()]
    pluralities = [p["plurality"] for p in per_prompt.values()]
    early = [p["early_exp_share"] for p in per_prompt.values()]
    late = [p["late_exp_share"] for p in per_prompt.values()]
    pooled = {
        "exp_frac_mean": float(np.mean(exp_fracs)),
        "exp_frac_min": float(np.min(exp_fracs)),
        "exp_frac_max": float(np.max(exp_fracs)),
        "exp_frac_std": float(np.std(exp_fracs)),
        "plurality_set": sorted(set(pluralities)),
        "plurality_stable": len(set(pluralities)) == 1,
        "mean_early_exp_share": float(np.mean(early)),
        "mean_late_exp_share": float(np.mean(late)),
        "depth_strengthens": float(np.mean(late)) > float(np.mean(early)),
        "n_prompts": len(per_prompt),
    }
    return {"model": model_name, "nhead": nhead, "per_prompt": per_prompt, "pooled": pooled}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gpt2", "EleutherAI/pythia-160m", "facebook/opt-125m"])
    ap.add_argument("--repeat", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    np.random.seed(SEED)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import torch  # noqa: F401
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"INCONCLUSIVE: deps unavailable ({type(e).__name__}: {e}).")
        return

    print(f"Tail robustness suite | {len(PROMPTS)} distinct prompts x {len(args.models)} models "
          f"| per-head KS | seed {SEED}\n")
    all_results = {}
    for mname in args.models:
        print(f"=== {mname} ===")
        r = run_model(mname, args.repeat, args.max_len)
        if r is None:
            continue
        all_results[mname] = r
        # per-prompt table
        print(f"  {'prompt':>10} | {'exp':>4} {'gauss':>5} {'unif':>4} | {'exp%':>5} | plurality")
        for pname, p in r["per_prompt"].items():
            c = p["tally"]
            print(f"  {pname:>10} | {c.get('exponential',0):>4} {c.get('gaussian',0):>5} "
                  f"{c.get('uniform',0):>4} | {p['exp_frac']*100:>4.0f}% | {p['plurality']}")
        pooled = r["pooled"]
        if pooled:
            print(f"  -> exp share across prompts: mean {pooled['exp_frac_mean']*100:.0f}%  "
                  f"range [{pooled['exp_frac_min']*100:.0f}%, {pooled['exp_frac_max']*100:.0f}%]  "
                  f"(std {pooled['exp_frac_std']*100:.0f} pts)")
            print(f"  -> plurality {'STABLE' if pooled['plurality_stable'] else 'VARIES'}: "
                  f"{pooled['plurality_set']}")
            print(f"  -> depth: mean exp share early {pooled['mean_early_exp_share']*100:.0f}% vs "
                  f"late {pooled['mean_late_exp_share']*100:.0f}%  "
                  f"(strengthens with depth: {pooled['depth_strengthens']})")
        print()

    # cross-model headline
    print("=== CROSS-MODEL SUMMARY ===")
    for mname, r in all_results.items():
        p = r["pooled"]
        if not p:
            print(f"  {mname}: no usable prompts")
            continue
        print(f"  {mname:>26}: plurality {p['plurality_set']} "
              f"({'stable' if p['plurality_stable'] else 'varies'}); "
              f"exp {p['exp_frac_mean']*100:.0f}% +/-{p['exp_frac_std']*100:.0f}; "
              f"depth-strengthens={p['depth_strengthens']}")
    distinct = {tuple(r["pooled"]["plurality_set"]) for r in all_results.values() if r["pooled"]}
    print(f"\n  distinct plurality regimes observed across models: {sorted(distinct)}")
    any_depth = any(r["pooled"]["depth_strengthens"] for r in all_results.values() if r["pooled"])
    print(f"  'exponential strengthens with depth' on ANY model: {any_depth}  "
          f"(claim is refuted if False everywhere)")

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "tail_robustness_suite.json"
    with open(path, "w") as fh:
        json.dump({"seed": SEED, "prompts": list(PROMPTS), "results": all_results}, fh, indent=2)
    print(f"\n  raw results saved: {path}")


if __name__ == "__main__":
    main()
