"""Publication-quality figures for "Margin-Adaptive Retrieval Complexity".

WHAT THIS PRODUCES (300 DPI PDFs, mathtext labels, tight layout) in figures/out/:
  fig_separation_overview.pdf  -- softmax's flat N-read cost vs the max head's certified reads as a
                                  function of the gap Gamma, at several N. (New: not in the paper.)
  fig_tail_trichotomy_fit.pdf  -- E[Delta_N] vs N for Exp/Gauss/Uniform with closed-form fit
                                  overlays and R^2 / relative-error labels (Thm 5).
  fig_precision_fragility.pdf  -- log10(required dynamic range) vs N for the super-increasing
                                  construction of Prop 4: it grows like N, so the construction is
                                  infeasible in fixed precision. (Honesty figure missing from paper.)
  fig_real_model_score_tails.pdf -- GPT-2 small attention logits on a fixed prompt; per (layer,head)
                                  we fit Exp/Gaussian/Uniform to the top-gap and report which tail
                                  regime dominates. (New: connects Thm 5 to a real model.) Skips
                                  gracefully (writes a note panel) if transformers/torch unavailable.

PASS = all available figures render to PDF without error; the real-model panel either renders or
records a clear 'unavailable' note. Reproducible with seed 42.
Run:  python figures/make_all.py
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.titlesize": 12, "mathtext.fontset": "cm",
})
C = {"exp": "#1b9e77", "gauss": "#7570b3", "unif": "#d95f02", "soft": "#d95f02", "max": "#1b9e77"}


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")


# ---------------------------------------------------------------------------
def fig_separation_overview(rng):
    """Softmax pays N (flat) reads; max head certifies the winner from a gap-dependent prefix."""
    def certified_reads(scores_desc, c):
        best = -np.inf
        for r in range(len(scores_desc)):
            best = max(best, scores_desc[r])
            nxt = scores_desc[r + 1] if r + 1 < len(scores_desc) else -np.inf
            if best >= c * max(nxt, 1e-12):
                return r + 1
        return len(scores_desc)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    Gammas = np.linspace(1.02, 6.0, 24)
    for N, col in [(256, "#9ecae1"), (4096, "#4292c6"), (65536, "#08519c")]:
        frac = []
        for G in Gammas:
            rr = []
            for _ in range(80):
                base = np.abs(rng.standard_normal(N - 1)) + 1.0
                s = np.concatenate([[base.max() * G], base])
                rr.append(certified_reads(np.sort(s)[::-1], c=1.05))
            frac.append(np.mean(rr) / N)
        ax.plot(Gammas, frac, "o-", color=col, ms=4, lw=2, label=fr"max head, $N=2^{{{int(np.log2(N))}}}$")
    ax.axhline(1.0, color=C["soft"], lw=2.5, ls="--", label=r"softmax (reads all $N$, every gap)")
    ax.set_yscale("log")
    ax.set_xlabel(r"multiplicative gap $\Gamma = s_\ast / s_{(2)}$")
    ax.set_ylabel(r"fraction of records read $\;/\,N$")
    ax.set_title("The separation: softmax is flat, max is gap-adaptive")
    ax.legend(loc="center right", fontsize=9)
    fig.savefig(OUT / "fig_separation_overview.pdf")
    plt.close(fig)
    return "fig_separation_overview.pdf"


def fig_tail_trichotomy_fit(rng):
    Ns = np.array([2 ** k for k in range(6, 17)])
    reps = lambda N: int(np.clip(2_000_000 // N, 300, 4000))

    def gap(samp, N):
        X = samp((reps(N), N)); X.sort(axis=1)
        return float((X[:, -1] - X[:, -2]).mean())

    laws = {
        "Exp(1)": (lambda sh: rng.exponential(1, sh), lambda N: np.ones_like(N, float),
                   r"$\mathbb{E}[\Delta_N]=a$ (const)", C["exp"]),
        "N(0,1)": (lambda sh: rng.standard_normal(sh), lambda N: 1 / np.sqrt(2 * np.log(N)),
                   r"$a/\sqrt{2\ln N}$", C["gauss"]),
        "U[0,1]": (lambda sh: rng.uniform(0, 1, sh), lambda N: 1 / (N + 1.0),
                   r"$a/(N{+}1)$", C["unif"]),
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for name, (samp, law, lab, col) in laws.items():
        emp = np.array([gap(samp, int(N)) for N in Ns])
        pred = law(Ns)
        a = np.sum(emp * pred) / np.sum(pred * pred)
        rel = np.mean(np.abs(emp - a * pred) / emp)
        ax.plot(Ns, emp, "o", color=col, ms=6, mfc="white", mew=1.5)
        ax.plot(Ns, a * pred, "-", color=col, lw=2,
                label=fr"{name}: {lab},  rel.err={rel:.2%}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(r"context length $N$")
    ax.set_ylabel(r"expected top gap $\mathbb{E}[\Delta_N]$")
    ax.set_title(r"Tail trichotomy (Thm 5): lines theory, $\circ$ simulation")
    ax.legend(loc="lower left", fontsize=9)
    fig.savefig(OUT / "fig_tail_trichotomy_fit.pdf")
    plt.close(fig)
    return "fig_tail_trichotomy_fit.pdf"


def fig_precision_fragility():
    """Super-increasing a_j > sum_{l<j} a_l forces a_j >= 2^{j-1}; the dynamic range (max/min) and
    the bits needed grow like N. This is why the exact-construction proof of Prop 4 is infeasible in
    fixed precision -- and why the patched proof (Augmented Indexing) is needed."""
    Ns = np.arange(8, 260, 4)
    # super-increasing minimal sequence: a_1=1, a_j = 1 + sum_{l<j} a_l = 2^{j-1}
    log10_range = (Ns - 1) * np.log10(2.0)  # log10(a_N / a_1) = (N-1) log10 2
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(Ns, log10_range, "-", color="#c0392b", lw=2.2, label=r"super-increasing: $\log_{10}(a_N/a_1)=(N{-}1)\log_{10}2$")
    ax.axhline(np.log10(2) * 53, color="#2c3e50", ls="--", lw=1.5, label=r"float64 mantissa limit ($\sim$53 bits)")
    ax.axhline(np.log10(2) * 24, color="#7f8c8d", ls=":", lw=1.5, label=r"float32 mantissa limit ($\sim$24 bits)")
    # mark where it breaks
    nbreak64 = 1 + 53
    ax.axvline(nbreak64, color="#2c3e50", ls="--", lw=0.8, alpha=0.6)
    ax.annotate(f"breaks at $N\\approx{nbreak64}$", xy=(nbreak64, np.log10(2) * 53),
                xytext=(nbreak64 + 12, np.log10(2) * 53 + 5), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#2c3e50"))
    ax.set_xlabel(r"number of values $N$")
    ax.set_ylabel(r"required dynamic range $\log_{10}(a_N/a_1)$")
    ax.set_title("Precision fragility of the exact Prop 4 construction")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(OUT / "fig_precision_fragility.pdf")
    plt.close(fig)
    return "fig_precision_fragility.pdf"


def _fit_tail_regime(gaps_samples: np.ndarray) -> str:
    """Crude tail classifier for a sample of top-gaps: compare coefficient of variation to the
    three reference regimes. Exponential spacing has CV=1 (mean=std); bounded/uniform gaps are
    tightly concentrated (CV small); Gaussian-tail gaps are intermediate. Returns a label."""
    m, sd = gaps_samples.mean(), gaps_samples.std()
    if m <= 0:
        return "degenerate"
    cv = sd / m
    if cv > 0.75:
        return "heavy/exponential-like"
    if cv < 0.35:
        return "bounded/light"
    return "Gaussian-like"


def fig_real_model_score_tails():
    """GPT-2 small: extract attention logits (pre-softmax scores) on a fixed prompt, per layer/head;
    classify the top-gap tail regime. Connects Thm 5 to a real model. Skips gracefully if deps
    are missing or the model cannot be downloaded (offline)."""
    note = None
    try:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
        model.eval()
        prompt = ("The quick brown fox jumps over the lazy dog. In a distant kingdom, a young "
                  "scholar discovered an ancient theorem that would change mathematics forever.")
        ids = tok(prompt, return_tensors="pt")
        # Pull hidden states, then reconstruct the pre-softmax QK^T/sqrt(dh) logits per layer/head
        # by replaying each block's own LayerNorm + c_attn projection. Everything under no_grad.
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
            hs = out.hidden_states  # tuple (L+1) of [1, T, d]
            L = len(model.transformer.h)
            nhead = model.config.n_head
            d = model.config.n_embd
            dh = d // nhead
            regimes = {}
            gap_samples_per_layer = []
            for li, blk in enumerate(model.transformer.h):
                x = blk.ln_1(hs[li])                 # block input -> LayerNorm
                qkv = blk.attn.c_attn(x)             # [1, T, 3d]
                q, k, _v = qkv.split(d, dim=2)
                T = q.shape[1]
                q = q.view(1, T, nhead, dh).permute(0, 2, 1, 3)
                k = k.view(1, T, nhead, dh).permute(0, 2, 1, 3)
                logits = ((q @ k.transpose(-1, -2)) / np.sqrt(dh))[0].numpy()  # [head, Tq, Tk]
                # causal: each query pos sees its prefix; collect the top-gap per (head, qpos>=2)
                layer_gaps = []
                for hh in range(nhead):
                    for qp in range(2, T):
                        row = np.sort(logits[hh, qp, : qp + 1])
                        layer_gaps.append(row[-1] - row[-2])
                layer_gaps = np.array(layer_gaps)
                gap_samples_per_layer.append(layer_gaps)
                regimes[li] = _fit_tail_regime(layer_gaps)
        # plot: heatmap-ish bar of mean top-gap per layer + regime annotation
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.4))
        means = [g.mean() for g in gap_samples_per_layer]
        cvs = [g.std() / g.mean() for g in gap_samples_per_layer]
        axA.bar(range(L), means, color="#4292c6")
        axA.set_xlabel("layer"); axA.set_ylabel(r"mean top-gap $\overline{\Delta}$ (logits)")
        axA.set_title("GPT-2 small: attention score top-gap by layer")
        axB.bar(range(L), cvs, color="#e67e22")
        axB.axhline(1.0, color="#1b9e77", ls="--", lw=1.2, label="CV=1 (exponential tail)")
        axB.axhline(0.35, color="#7f8c8d", ls=":", lw=1.2, label="CV=0.35 (bounded threshold)")
        axB.set_xlabel("layer"); axB.set_ylabel("coeff. of variation of top-gap")
        axB.set_title("Tail regime per layer (CV-based)")
        axB.legend(fontsize=9)
        fig.suptitle("Real-model score tails (GPT-2 small): are gaps heavy-tailed in practice?",
                     fontsize=12)
        fig.savefig(OUT / "fig_real_model_score_tails.pdf")
        plt.close(fig)
        # text summary
        from collections import Counter
        cnt = Counter(regimes.values())
        print("  GPT-2 per-layer tail regimes:", dict(cnt))
        return "fig_real_model_score_tails.pdf"
    except Exception as e:  # offline / missing model -> graceful note panel
        note = f"GPT-2 figure unavailable: {type(e).__name__}: {str(e)[:120]}"
        fig, ax = plt.subplots(figsize=(6.6, 3.0))
        ax.axis("off")
        ax.text(0.5, 0.5, note + "\n(run online once to cache gpt2, then re-run)",
                ha="center", va="center", wrap=True, fontsize=10)
        fig.savefig(OUT / "fig_real_model_score_tails.pdf")
        plt.close(fig)
        print("  " + note)
        return "fig_real_model_score_tails.pdf (NOTE panel)"


def fig_learned_rescaling_phase(rng):
    """Cor 10.2 phase diagram: per-layer cost exponent rho^(l) vs layer l for a grid of (alpha,gamma)
    pairs, with the alpha*gamma=1 boundary marked. Below the boundary (alpha*gamma<1) the margin
    decays and rho->1; on it, rho stays flat."""
    N, d, L = 256, 32, 12
    a_lsh = 1.0

    def contraction(g):
        J = np.ones((N, N)) / N
        return (1 - g) * J + g * np.eye(N)

    def rho_curve(gamma, alpha, reps=120):
        W = contraction(gamma)
        rhos = np.zeros((reps, L + 1))
        for r in range(reps):
            X = rng.standard_normal((N, d)); X -= X.mean(0, keepdims=True)
            qd = rng.standard_normal(d)

            def gap(M):
                s = np.sort((M @ qd) / np.sqrt(d))
                return s[-1] - s[-2]
            rhos[r, 0] = 1.0 / (1.0 + a_lsh * gap(X))
            for l in range(1, L + 1):
                X = alpha * (W @ X); X -= X.mean(0, keepdims=True)
                rhos[r, l] = 1.0 / (1.0 + a_lsh * gap(X))
        return rhos.mean(0)

    layers = np.arange(L + 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    pairs = [(0.7, 1.0), (0.7, 1.2), (0.9, 1.05), (0.7, 1.0 / 0.7), (0.5, 2.0)]
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(pairs)))
    for (gamma, alpha), c in zip(pairs, cmap):
        ag = alpha * gamma
        style = "--" if abs(ag - 1.0) < 1e-6 else "-"
        ax.plot(layers, rho_curve(gamma, alpha), style, color=c, lw=2,
                label=fr"$\gamma={gamma:.2f},\ \alpha={alpha:.2f}$ ($\alpha\gamma={ag:.2f}$)")
    ax.axhline(1.0, color="gray", ls=":", lw=1.2, label=r"linear scan ($\rho=1$)")
    ax.set_xlabel(r"layer $\ell$")
    ax.set_ylabel(r"per-layer cost exponent $\rho^{(\ell)}$")
    ax.set_title(r"Learned-rescaling phase diagram: $\alpha\gamma=1$ is the boundary")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(OUT / "fig_learned_rescaling_phase.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return "fig_learned_rescaling_phase.pdf"


def fig_skippability_surface(rng):
    """Prop 4.3 surface: heatmap of skippable fraction vs (eps, Delta) at N=1024."""
    N, dv = 1024, 8

    def skip_frac(margin, eps, reps=150):
        fr = []
        for _ in range(reps):
            s = rng.standard_normal(N)
            o = np.argsort(s)
            s[o[-1]] = s[o[-2]] + margin
            z = s - s.max(); w = np.exp(z); w /= w.sum()
            V = rng.standard_normal((N, dv))
            fr.append(np.mean(w * np.linalg.norm(V, axis=1) <= eps))
        return np.mean(fr)

    margins = np.linspace(0, 8, 9)
    epss = np.linspace(0.001, 0.2, 9)
    grid = np.array([[skip_frac(m, e) for e in epss] for m in margins])
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                   extent=[epss[0], epss[-1], margins[0], margins[-1]], vmin=0, vmax=1)
    ax.set_xlabel(r"error budget $\varepsilon$")
    ax.set_ylabel(r"score margin $\Delta$")
    ax.set_title(r"$\varepsilon$-skippable fraction (Prop 4.3, $N=1024$)")
    fig.colorbar(im, ax=ax, label="fraction skippable")
    fig.savefig(OUT / "fig_skippability_surface.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return "fig_skippability_surface.pdf"


def fig_cancellation_counterexample():
    """Prop 4.3 necessity is singleton-only: two contributions w_j v_j each of norm > eps whose
    SUM has norm < eps, so {j1,j2} is a valid skip set even though neither is skippable alone."""
    eps = 0.05
    w = 0.04
    # two value vectors at a slight angle so they nearly (not exactly) cancel
    ang = np.deg2rad(170)  # almost opposite
    mag = (eps / w) * 1.1  # ||w v|| = 1.1 eps > eps individually
    v1 = mag * np.array([1.0, 0.0])
    v2 = mag * np.array([np.cos(ang), np.sin(ang)])
    c1, c2 = w * v1, w * v2
    s = c1 + c2
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    # circle of radius eps
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(eps * np.cos(th), eps * np.sin(th), "k--", lw=1, label=fr"$\|\cdot\|=\varepsilon={eps}$")
    ax.quiver(0, 0, c1[0], c1[1], angles="xy", scale_units="xy", scale=1, color="#1b9e77",
              width=0.012, label=fr"$w_1 v_1$ ($\|\cdot\|={np.linalg.norm(c1):.3f}>\varepsilon$)")
    ax.quiver(c1[0], c1[1], c2[0], c2[1], angles="xy", scale_units="xy", scale=1, color="#7570b3",
              width=0.012, label=fr"$w_2 v_2$ ($\|\cdot\|={np.linalg.norm(c2):.3f}>\varepsilon$)")
    ax.quiver(0, 0, s[0], s[1], angles="xy", scale_units="xy", scale=1, color="#d95f02",
              width=0.016, label=fr"sum ($\|\cdot\|={np.linalg.norm(s):.3f}\leq\varepsilon$)")
    lim = mag * w * 1.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.9, lim * 0.9)
    ax.set_aspect("equal"); ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
    ax.set_title(r"Prop 4.3: necessity is singleton-only ($|R|{=}2$ cancellation)")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(OUT / "fig_cancellation_counterexample.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return "fig_cancellation_counterexample.pdf"


def fig_sandwich(rng):
    """Prop 9.1 sandwich: both bounds and the true w_* vs N, showing the RANGE bound is binding."""
    R, tau, Delta = 4.0, 1.0, 2.0
    Ns = np.array([2 ** k for k in range(1, 17)])
    lo = 1.0 / (1.0 + (Ns - 1) * np.exp(-Delta / tau))     # margin lower bound
    up = 1.0 / (1.0 + (Ns - 1) * np.exp(-2 * R / tau))     # range upper bound (the ceiling)
    # empirical true w_* averaged over random bounded-score instances at each N
    true = []
    for N in Ns:
        ws = []
        for _ in range(400):
            s_star = rng.uniform(0, R)
            hi = s_star - Delta
            if hi < -R:
                ws.append(np.nan); continue
            others = rng.uniform(-R, hi, N - 1)
            sc = np.concatenate([[s_star], others])
            z = sc - sc.max(); w = np.exp(z / tau); w /= w.sum()
            ws.append(w[0])
        true.append(np.nanmean(ws))
    true = np.array(true)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(Ns, up, "-", color="#d95f02", lw=2.2, label=r"range upper bound $\frac{1}{1+(N-1)e^{-2R/\tau}}$ (ceiling)")
    ax.plot(Ns, true, "o", color="#1f78b4", ms=5, mfc="white", mew=1.3, label=r"true $w_\ast$ (sim)")
    ax.plot(Ns, lo, "-", color="#1b9e77", lw=2.2, label=r"margin lower bound $\frac{1}{1+(N-1)e^{-\Delta/\tau}}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(r"context length $N$"); ax.set_ylabel(r"winner weight $w_\ast$")
    ax.set_title(r"Prop 9.1 sandwich: the range bound is the binding ceiling ($R{=}4,\Delta{=}2$)")
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(OUT / "fig_sandwich.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return "fig_sandwich.pdf"


def main():
    rng = np.random.default_rng(SEED)
    made = []
    made.append(fig_cancellation_counterexample())
    made.append(fig_sandwich(np.random.default_rng(SEED)))
    made.append(fig_skippability_surface(np.random.default_rng(SEED)))
    made.append(fig_precision_fragility())
    made.append(fig_learned_rescaling_phase(np.random.default_rng(SEED)))
    made.append(fig_tail_trichotomy_fit(np.random.default_rng(SEED)))
    made.append(fig_separation_overview(np.random.default_rng(SEED)))
    made.append(fig_real_model_score_tails())
    print("\nwrote to", OUT)
    for m in made:
        print("  -", m)


if __name__ == "__main__":
    main()
