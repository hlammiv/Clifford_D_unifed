# Phase D implementation note: lazy DB population

When the U-net builder (or `sk_leaves.synthesize_leaf` rule 4) calls
`RzLookupDB.lookup(theta, eps_max)` and gets `None`, the builder MUST:

1. Fall back to a live HRSA or zeta9 synthesis for that (θ, ε_target).
2. **Insert the result back into the DB** via `RzLookupDB.insert(...)`.
3. Commit immediately so concurrent workers see it.

```python
result = rz_db.lookup(theta, eps_max)
if result is None:
    log_miss(theta, eps_max)
    result = compute_via_hrsa_or_zeta9(theta, eps_target)
    rz_db.insert(theta, eps_target, V=result.V, v_f=result.v_f,
                 achieved_frob=result.achieved_frob, N_D=result.N_D,
                 method=result.method, source="live_fallback")
    rz_db.commit()
return result
```

## Why

Pre-populating the DB at tight ε is *very* expensive:
- ε=10⁻³ needs ~210k stored angles for Euler-decomp leaf budgets
- ε=10⁻⁴ needs ~2.1M

Building U-net targets only ever exercises a tiny subset of those grids
(only the actual θ values requested by `euler_decompose(U)` for sampled
U's). Let the DB densify reactively where SK actually touches —
saves ~99% of the precompute work.

## Instrumentation requirement

The builder MUST log the miss rate per ε_target tier:

```
[u_net_builder] tier eps=1e-3: 4847 lookups, 312 misses (6.4%), 312 live calls in 18234s
```

If miss rate stays high (>20%) the strategy should escalate to bulk
pre-population for that tier. <5% means lazy is winning.

## Cross-reference

Memory: `sk_rz_db_lazy_population.md` (2026-05-23).

## Per-leaf budget: `ε_leaf = ε_U / √n_leaves`

`u_net_builder._reify_factors` splits the U-net error budget across the ~57
Z-leaves emitted by Euler decomposition. The original Phase D design used
`ε_leaf = ε_U` and relied on a post-product `1.5·ε_U` slack check.

Empirically that fails for arbitrary Haar targets (verified 2026-05-23, see
memory `tier0_slack_design_dead_2026-05-23`): the L2 error estimate
`ε_total ≈ ε_leaf · √n_leaves` is ~7.5× over the slack ceiling at any ε_U.

The current design uses the L2-tight split:

```python
eps_leaf = max(float(eps_u) / max(1.0, float(n_z_leaves) ** 0.5), 1e-12)
```

If you change this back, expect 0/N yield on Haar samples with a densified DB.

## Hidden prereq: `/tmp/e0_net_closed.txt`

`hrsa/sk_leaves.synthesize_leaf` requires the 8100-element sign-extended
Clifford net at `/tmp/e0_net_closed.txt`. Rule 3 loads it before rule 4
even fires, so without it the entire lazy-fallback path raises
`FileNotFoundError`.

As of commit `285c025` (2026-05-23), `hrsa/ep_descent.load_e0_net()`
auto-builds the file on first import by invoking `hrsa/e0_net_dump`.
If the C++ binary is also missing: `cd hrsa && make e0_net_dump`.
