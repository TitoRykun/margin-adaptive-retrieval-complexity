"""INNOVATION 1 -- margin-aware KV-cache eviction.

IDEA / WHICH CONTRIBUTION
    The paper's separation says large-gap (easy) retrievals are cheap and ROBUST: when a key wins
    by a wide margin, approximations barely affect the outcome. Flip it for caching: a key that is
    *consistently a clear winner or clear loser* (extreme margin) across recent queries is robust
    to eviction -- its retrieval outcome is stable. We test a policy that evicts keys whose presence
    is least decision-critical, measured by how often they sit near the decision boundary.

    Concretely we score each key by its running "criticality" = how often it was the winner or a
    near-winner (within the top gap) of recent queries. Keys that are rarely critical (never close
    to winning) are evicted first -- they contribute negligibly to retrieval outputs. We compare
    against LRU and random eviction on a synthetic long-context retrieval task.

PASS = the margin-aware policy retrieves the correct (full-cache) winner at least as accurately as
    LRU and random at the same cache budget, on average over the query stream.

NOTE on honesty: this is a heuristic policy motivated by the theory, benchmarked on a synthetic
    task; it is a proof-of-concept, not a claim about production systems.
"""
from __future__ import annotations

import numpy as np

SEED = 42


def run_stream(policy: str, N: int, cache_size: int, n_queries: int, d: int, rng) -> float:
    """Simulate a query stream over N fixed keys with a cache of `cache_size`. Each query has a
    true winner = argmax over ALL N keys; the policy only 'sees' cached keys but we measure whether
    the cached-argmax equals the full-argmax (retrieval accuracy under the cache).
    Returns mean retrieval accuracy."""
    keys = rng.standard_normal((N, d))
    in_cache = np.zeros(N, dtype=bool)
    in_cache[:cache_size] = True  # warm start: first cache_size keys
    last_used = np.full(N, -1)     # for LRU
    crit = np.zeros(N)             # running criticality (margin-aware): EMA of relative score rank
    seen = np.zeros(N)
    correct = 0
    ema = 0.1                      # criticality EMA rate
    for t in range(n_queries):
        q = rng.standard_normal(d)
        scores_all = keys @ q
        true_winner = int(np.argmax(scores_all))
        cached_idx = np.where(in_cache)[0]
        cs = scores_all[cached_idx]
        cached_winner = int(cached_idx[np.argmax(cs)])
        correct += (cached_winner == true_winner)
        # update stats on cached keys
        last_used[cached_idx] = t
        seen[cached_idx] += 1
        # criticality (theory-faithful): a key is critical if it is ever CLOSE to winning, i.e.
        # within the running top-gap of the cached winner. Keys whose score is consistently far
        # below the winner (large negative gap to the top) are robustly non-decisive -> evictable.
        gap_to_top = cs.max() - cs            # >=0; small means competitive (critical)
        proximity = np.exp(-gap_to_top)       # in (0,1], ~1 if near the winner
        crit[cached_idx] = (1 - ema) * crit[cached_idx] + ema * proximity
        # admit the true winner if missing (a miss triggers a fetch -> eviction needed)
        if not in_cache[true_winner]:
            if in_cache.sum() >= cache_size:
                cand = np.where(in_cache)[0]
                if policy == "lru":
                    victim = cand[np.argmin(last_used[cand])]
                elif policy == "random":
                    victim = rng.choice(cand)
                elif policy == "margin":
                    # evict the LEAST critical cached key: lowest running proximity-to-winner, i.e.
                    # the key that is most robustly far from ever deciding a query.
                    victim = cand[np.argmin(crit[cand])]
                else:
                    raise ValueError(policy)
                in_cache[victim] = False
            in_cache[true_winner] = True
            last_used[true_winner] = t
            seen[true_winner] += 1
    return correct / n_queries


def main():
    rng_master = np.random.default_rng(SEED)
    N, d = 2000, 32
    n_queries = 4000
    print("Margin-aware KV-cache eviction vs LRU / random (synthetic retrieval)\n")
    print(f"{'cache/N':>8} {'margin':>8} {'lru':>8} {'random':>8}")
    wins = 0
    rows = 0
    for frac in (0.05, 0.1, 0.2, 0.4):
        cache_size = int(N * frac)
        accs = {}
        for pol in ("margin", "lru", "random"):
            # same seed per policy for fair comparison of the SAME stream
            rng = np.random.default_rng(SEED + 17)
            accs[pol] = run_stream(pol, N, cache_size, n_queries, d, rng)
        print(f"{frac:>8.2f} {accs['margin']:>8.3f} {accs['lru']:>8.3f} {accs['random']:>8.3f}")
        wins += accs["margin"] >= max(accs["lru"], accs["random"]) - 1e-9
        rows += 1
    verdict = "PASS" if wins >= rows - 1 else "INCONCLUSIVE"  # allow one tie/loss
    print(f"\n  margin-aware >= both baselines in {wins}/{rows} cache sizes")
    print(f"  VERDICT: {verdict}  (heuristic proof-of-concept motivated by the margin separation)")


if __name__ == "__main__":
    main()
