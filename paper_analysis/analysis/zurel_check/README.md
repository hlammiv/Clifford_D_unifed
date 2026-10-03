# Zurel–Jana–de Silva QQR codes vs. the R-state (Golay-type) route

Source: M. Zurel, S. Jana, N. de Silva, "High-threshold magic state distillation with quantum
quadratic residue codes", arXiv:2603.18560v2 (30 Sep 2026), QST 11, 045052. The full PDF (42 pp.)
was fetched from arxiv.org/pdf/2603.18560 and read. Page numbers below are the paper's printed pages.

Files
- `zurel_codes.py`: the qutrit weight enumerators transcribed from **Table 5 (App. C, p. 37)**.
  Each one passes the size check sum A_w = 9^((p-1)/2).
- `zurel_check.py`: imports `unified/factory_model/factory_cost.py` read-only. It writes no
  bytecode and copies nothing. It adds the QQR codes as alternative strange-state codes.
- `zurel_check_output.txt`: the full output.

Run it with `python3 zurel_check.py`, which takes about 4 s.

## What the paper provides for qutrits

- All QQR codes are `[[p,1,d]]_3` with p ≡ 5, 11 (mod 12). They encode one qutrit, and the
  protocol is n→1 (Sec. 3.2, p. 10; Table 1, p. 11).
- The 11-qutrit QQR code is the ternary-Golay code of Prakash (Sec. 4.1, p. 15).
- Target state: the Strange state |S⟩ = (|1⟩−|2⟩)/√2 (p. 23). This is the same state as `S` in
  `conversions.py`, so the outputs feed the SS→N→R-state conversion of our model unchanged.
- Noise model: ρ_S(ε) = (1−ε)|S⟩⟨S| + ε·1/3 (Eq. 107, p. 24). ε is a depolarising parameter,
  and infidelity = 2ε/3. This is the `delta` of the model's `golay_map`.
- Distillation map: ε′(ε) = ¾[1 + 3 W_I(x,y)/W_{I⊥}(x,y)], with x = (4ε−3)/9 and y = (3−ε)/18
  (Eqs. 108, 118–120, pp. 24–25).
- Success probability: P_s = W_{I⊥}(x,y) (Eq. 121, p. 25). This is post-selection on a **single**
  (trivial) syndrome. The paper gives no error-corrected or multi-syndrome variant. App. E,
  p. 42, mentions one only as future work.
- The paper gives no closed-form leading coefficients and no yields. It gives thresholds
  (Table 3, p. 26), plots (Figs. 3–4, p. 27; Fig. 6, p. 38) and the weight enumerators (Table 5,
  p. 37). The coefficients c and m below are my evaluation of Eqs. 120–121 from Table 5.

Checks:
- The computed thresholds reproduce Table 3 to every printed digit.
- For p = 11 the map agrees with the model's Prakash `golay_map` to 1e-15 in ε_out.
- For p = 11, P_s(0) = 1/1728, also matching the model.
- The model's Golay P_s is first-order linear. The exact P_s differs from it by only ~0.3% at
  p = 1e-2, so this does not matter.

### Protocol table

ε is the depolarising parameter, ε_out ≈ c·ε^m, and P_s is the probability of accepting one
output.

| code | [[n,k,d]]_3 | inputs/output (accepted) | threshold ε* (Table 3, p.26) | m | c | P_s(ε→0) | P_s at p=1e-2 (ε=.015) | raw inputs per output, n/P_s | output |
|---|---|---|---|---|---|---|---|---|---|
| QQR11 = Golay (Prakash) | [[11,1,5]] | 11 | 0.38715 | 3 | 3.06 | 5.79e-4 (=1/1728) | 5.18e-4 | 1.9e4 | strange state |
| QQR17 (new) | [[17,1,7]] | 17 | 0.34394 | **1** | 0.0236 | 3.36e-6 | 2.83e-6 | 5.1e6 | strange state |
| QQR23 (new) | [[23,1,8]] | 23 | 0.16636 | 3 | 14.1 | 7.75e-10 | 6.15e-10 | 3.0e10 | strange state |
| QQR41 (new) | [[41,1,13]] | 41 | 0.31877 | **1** | 4.59e-3 | 3.15e-15 | 2.09e-15 | 1.3e16 | strange state |
| QQR47 (new) | [[47,1,14]] | 47 | 0.04864 | 3 | 60.1 | 3.48e-22 | 2.17e-22 | 1.4e23 | strange state |
| QQR5, QQR29 | [[5,1,3]], [[29,1,11]] | – | none (fixed point only; Fig. 6) | 1 | – | – | – | – | – |

The new codes have high thresholds but are not better distillers in the sense that matters here:
- **QQR17 and QQR41 suppress error only linearly** (m = 1).
- QQR23 and QQR47 are cubic, like Golay, but with larger coefficients.
- Acceptance falls by roughly 3–4 orders of magnitude for every 6 extra qutrits.

The authors reach the same conclusion for qubits (App. E, pp. 39–42, Fig. 8): larger QR
protocols lose to concatenating small ones once acceptance is counted.

## Recomputed c_R/c_T (Golay-type row)

Setup: the same grid and budgets as `factory_cost.main` (budget 1e-2, 227 T and 111 R per
rotation). c_T is QRM_3(2) 8→1. The R cost is computed as

c_R = 3 (RUS injection) × 32 (SS→N→R-state conversion) × min over level sequences (≤ 5 levels,
any mix of codes) of Π n/P_s.

The conversion step has ε_R = 2.667 ε_S.

| p | N_rot | c_T | model Golay | Golay + QQR17/23/41/47 (real P_s) | optimistic: exact ε′, P_s = 1 | optimistic: c·ε^m, P_s = 1 |
|---|---|---|---|---|---|---|
| 1e-2 | 1e2 | 69.5 | 5.6e8 | 5.6e8 (Golay, Golay) | 167 | 167 |
| 1e-2 | 1e3 | 556 | 7.0e7 | 7.0e7 | 20.9 | 20.9 |
| 1e-2 | 1e4 | 556 | 7.0e7 | 7.0e7 | 20.9 | 20.9 |
| 1e-3 | 1e2 | 64.5 | 2.9e4 | 2.9e4 (Golay) | 16.4 | 16.4 |
| 1e-3 | 1e3 | 64.5 | 2.9e4 | 2.9e4 | 16.4 | 16.4 |
| 1e-3 | 1e4 | 64.5 | 5.4e8 | 5.4e8 | 180 | 180 |
| 1e-4 | 1e2 | 8.0 | 2.3e5 | 2.3e5 (Golay) | 132 | 132 |
| 1e-4 | 1e3 | 8.0 | 2.3e5 | 2.3e5 | 132 | 132 |
| 1e-4 | 1e4 | 64.1 | 2.9e4 | 2.9e4 | 16.5 | 16.5 |

**Range with the Zurel codes added: c_R/c_T = 2.9e4 to 5.6e8.** This is identical to the
original model. The optimiser never chooses a QQR17/23/41/47 level, alone or mixed with Golay.

**Optimistic bound: c_R/c_T = 16 to 180.** Here acceptance is set to 1 for every code. The cheapest
choice is still Golay (QQR11), one or two levels. The full-map and leading-order versions agree.

The optimistic bound is set almost entirely by structure, not by the code. Each R gate needs at
least 96 raw strange states per accepted distilled strange state (3 for injection × 32 for
conversion), multiplied by at least 11 inputs per distillation level. Every new code is larger
than 11, so none of them can lower that bound.

An unphysical extra relaxation is to make the SS→N and NN→R conversions deterministic as well.
The model's 1/2 and 1/4 are the exact pure-state probabilities from `conversions.py`. With that
relaxation the ×32 becomes ×4, and the ratio becomes 2.0–22.5.

## Verdict

**No.** With the Zurel–Jana–de Silva QQR codes, the strange-state → R-state route does not come
anywhere near the break-even range c_R/c_T ≈ 1.4–2.6, nor the 7-T gadget value of 7.
- **With acceptance counted properly:** the new codes are never used. The Golay-type row stays
  at c_R/c_T = 3e4 to 6e8.
- **With every distillation round accepting with probability 1:** the best is still Golay, at
  c_R/c_T = 16 to 180. That is 2.3–26 times worse than the gadget's 7 at every grid point.
  The new codes cannot help there, because each has n > 11 and either linear suppression
  (17, 41) or larger cubic coefficients (23, 47).

The paper's improvements are in **threshold**, which is irrelevant for p ≤ 1e-2. Golay's
threshold is already ε* ≈ 0.39, i.e. infidelity ≈ 0.26. The paper improves nothing in
suppression order, coefficient or yield.
