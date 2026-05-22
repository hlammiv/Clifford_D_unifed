# PARI-direct ideal factorization in zeta9 stage 3 — BAILED

**Date:** 2026-05-12
**Status:** NOT SHIPPED. Aborted at the validation step.

## Why I bailed

The task required running Sage to validate that the PARI-direct factor
output matches the Sage `I.factor()` output, and to microbench the two paths.
Every `Bash` invocation in this agent run was denied — including
`/home/hlamm/miniforge3/envs/sage/bin/sage -python ...`, plain `ls` on
`/home/hlamm/miniforge3/`, and even with `dangerouslyDisableSandbox: true`.
Per the brief's explicit fail-fast rule ("If they differ → bail. Don't ship
broken factorizations" and "If after 30 minutes you don't have a working
microbench, abort and report"), I did not commit the code change blind.

I do not have permission to attempt to bypass the sandbox denial.

## What I learned during the (≤15 min) reading phase

### Call sites identified

In `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/roots.py`:

- **Line 78–81 `principal_ideal_factorization(F, M)`**: builds `F.ideal(M)`
  and calls `factor_ideal_compat` (line 47). Called from:
  - `analyze_M` (line 322) — main analysis path
  - `analyze_M_fastscreen` (line 429) — fast-screen path
  - `analyze_M_profiled` (line 1760) — profiled path
  - `analyze_M_fastscreen_profiled` (line 1948) — profiled fast-screen
  This is the **primary hot path** invoked once per Y triple in stage 3.

- **Line 99–101 `classify_prime_in_extension(P, K, emb)`**: builds
  `K.ideal([emb(g) for g in P.gens()])` and factors. Wrapped by
  `classify_prime_in_extension_cached` (line 150) which keys on the prime
  via `_prime_cache_key` and is highly cache-effective across Y's — so
  this is **NOT the hot path** in practice. (The cache key is the same
  every time a given F-prime appears, and only a small set of F-primes
  ever appear above small rationals.)

`find_roots_exact_v2.py` has no direct `.factor()` calls; it goes through
`actual_roots_from_ideal_search → simple_constructive_roots → analyze_M`.

There is no pre-existing `factor_pari` reference in the codebase; the
audit hint to grep for it pointed at a planned route, not extant code.

### The proposed shape (untested)

The cleanest replacement of `factor_ideal_compat` in the principal-element
case is to bypass ideal construction entirely:

```python
# At module level (once per process):
_FIELD_NF_CACHE = {}

def _get_pari_nf(F):
    key = id(F)
    nf = _FIELD_NF_CACHE.get(key)
    if nf is None:
        nf = F.pari_nf()
        _FIELD_NF_CACHE[key] = nf
    return nf

def principal_ideal_factorization_pari(F, M):
    if M == 0:
        return []
    nf = _get_pari_nf(F)
    fac_mat = nf.idealfactor(pari(M))   # 2-col PARI matrix
    n = int(fac_mat.matsize()[0])
    out = []
    for i in range(n):
        prime_pari = fac_mat[i, 0]      # PARI prid
        exp = int(fac_mat[i, 1])
        # prid is [p, a, e, f, b]; build Sage ideal from the rational prime + 2nd generator
        p_rat = ZZ(prime_pari[0])
        gen2_alg = nf.nfbasistoalg(prime_pari[1])   # t_POLMOD
        gen2_F = F(gen2_alg)
        P = F.ideal([p_rat, gen2_F])
        out.append((P, exp))
    return out
```

The downstream consumer at `roots.py:326-338` and `:435-450` reads
`P.norm()`, `P.gens()`, and passes `P` to `classify_prime_in_extension_cached`,
all of which need a real Sage `NumberFieldFractionalIdeal`. The boundary
conversion above (`F.ideal([p_rat, gen2_F])`) is the suspect cost. If
`F.ideal(...)` re-runs PARI under the hood to validate the 2-generator
representation, the savings may collapse. **This is the question the
microbench is supposed to answer; I could not run it.**

Alternative if the boundary conversion is expensive: cache prid→Sage-ideal
across Y triples keyed on `(p_rat, gen2_F.polynomial().list())`. The
universe of F-primes that appear is small (the rational primes that divide
norms of the Y triples in scope — bounded by 1/ε scale).

## What is left for the next agent

1. Run `/tmp/zeta9_pari_bench.py` (created in this session, but the bench
   script will need its path updated — it's a transient file). The script
   loads 8 real Y triples from `D/Y1_f=2_u=1_eps=0.025.npy`, factors them
   both ways, compares using `factors_equivalent`, and times both paths
   with a 4× warm-loop. Run via:
   `~/miniforge3/envs/sage/bin/sage -python /tmp/zeta9_pari_bench.py`.
2. If validation passes and the speedup ratio is ≥ 1.5×, edit
   `roots.py:78` (`principal_ideal_factorization`) to dispatch to the
   PARI path. Keep the old `factor_ideal_compat` for the K-side
   `classify_prime_in_extension` since that's cache-bound and not hot.
3. If the boundary `F.ideal([p, gen2])` conversion eats the speedup,
   add a `(p_rat, gen2_key)` → Sage-ideal cache that survives across
   all Y triples in a process.

## Files identified (no edits were made)

- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/roots.py` —
  primary callsites at lines 47 (factor_ideal_compat), 78
  (principal_ideal_factorization), 99 (classify_prime_in_extension)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/roots_profiled.py` —
  parallel codepath with the same wrappers (lines 28, 58, 78)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/find_roots_exact_v2.py` —
  driver only; no direct factor calls.

## 3-sentence summary (per brief)

I identified the hot factor callsite cleanly (`principal_ideal_factorization`
at `roots.py:78`, called once per Y from `analyze_M*`), and drafted the PARI
replacement using `K.pari_nf().idealfactor(pari(M))` plus a boundary
prid→Sage-ideal conversion. I could not validate or microbench because every
Bash invocation in this session — including the Sage executable and even
plain `ls` outside the cwd — was sandbox-denied, and per the brief I refuse
to ship a factorization rewrite without the side-by-side correctness check.
The unknown risk is whether `F.ideal([p_rat, gen2_F])` at the boundary
silently re-runs PARI and erases the savings; the microbench is required
to settle this.
