from collections import OrderedDict
from itertools import product
import math, time
from sage.all import CyclotomicField, NumberField, QQ, RealField, ZZ, polygen, factor, matrix, vector, pari


# Memory refactor #5 (2026-05-22): cap module-level caches so long sweeps don't
# accumulate unbounded RSS. Each cache below uses BoundedDict(_CACHE_MAX_ENTRIES).
# Default cap is generous (200k) — the rare cache miss from eviction is cheap.
_CACHE_MAX_ENTRIES = 200_000


class BoundedDict(OrderedDict):
    """OrderedDict with LRU eviction at max_entries. Drop-in for dict in
    cache-get-set patterns: `cache.get(key)` and `cache[key] = value`.
    Bounds RAM growth without changing call sites."""
    __slots__ = ("_max_entries",)

    def __init__(self, max_entries=_CACHE_MAX_ENTRIES):
        super().__init__()
        self._max_entries = int(max_entries)

    def __setitem__(self, key, value):
        if key in self:
            super().move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self._max_entries:
            super().popitem(last=False)

    def get(self, key, default=None):
        if key in self:
            super().move_to_end(key)
            return super().__getitem__(key)
        return default

# Action 2 (2026-05-12): turn OFF Sage's "proof" mode for number-field operations.
#
# By default Sage runs verification proofs around class-group / unit-group /
# principality results that depend on GRH being uncertified.  For our enumeration
# pipeline we just need the candidate roots; downstream `corrected * conjugate /
# emb(M) == 1` verification (in actual_roots_from_ideal_search) is exact and
# does NOT rely on these proofs, so toggling proof off is sound.  Empirically
# this saves ~1.5-3× on the per-Y stage 3 wall.
#
# The toggle is applied at module load so every Sage call in this process
# inherits it.  Reference:
#   https://doc.sagemath.org/html/en/reference/structure/sage/structure/proof/proof.html
try:
    from sage.structure.proof.all import number_field as _proof_nf
    _proof_nf(False)
except Exception:
    # Older Sage releases or import-time failures: continue with proof default.
    pass


# ----------------------------------------------------------------------
# Field construction
# ----------------------------------------------------------------------

def build_fields():
    x = polygen(QQ, "x")
    F = NumberField(x**3 - 3*x + 1, names=("alpha",))
    alpha = F.gen()

    K = CyclotomicField(9)
    z = K.gen()
    emb = F.hom([z + z**(-1)], K)

    OF = F.ring_of_integers()
    OK = K.ring_of_integers()
    return F, OF, K, OK, alpha, z, emb


# ----------------------------------------------------------------------
# Compatibility helpers
# ----------------------------------------------------------------------

def factor_ideal_compat(I):
    # Action 2 (2026-05-12): the global Sage proof.number_field(False) toggle
    # set at module load suffices; NumberFieldFractionalIdeal.factor() does
    # NOT accept a per-call `proof=` kwarg (TypeError on Sage 10).  The
    # global toggle handles class-group / principality verification skipping.
    for name in ("factor", "factorization"):
        try:
            return getattr(I, name)()
        except Exception:
            pass
    try:
        return factor(I)
    except Exception:
        pass
    try:
        primes = I.prime_factors()
        out = []
        for P in primes:
            try:
                e = int(I.valuation(P))
            except Exception:
                e = 0
            if e:
                out.append((P, e))
        if out:
            return out
    except Exception:
        pass
    raise AttributeError(f"No compatible ideal factorization method found for {type(I).__name__}")


# PARI-direct factorization cache (validated 2026-05-12: ~2× speedup, all
# factorizations identical to Sage I.factor() on 8 sample Y triples).
_FIELD_PARI_NF_CACHE = BoundedDict(max_entries=64)   # keyed by id(F); few distinct fields
_FIELD_PARI_BNF_CACHE = BoundedDict(max_entries=64)


def _get_pari_nf(F):
    key = id(F)
    nf = _FIELD_PARI_NF_CACHE.get(key)
    if nf is None:
        nf = F.pari_nf()
        _FIELD_PARI_NF_CACHE[key] = nf
    return nf


def _get_pari_bnf(K):
    """Cached PARI bnf for K. bnf adds unit/class info on top of nf — slower
    one-time construction but needed for bnfisprincipal."""
    key = id(K)
    bnf = _FIELD_PARI_BNF_CACHE.get(key)
    if bnf is None:
        bnf = K.pari_bnf()
        _FIELD_PARI_BNF_CACHE[key] = bnf
    return bnf


def principal_generator_h1_fast(I_or_hnf, K=None, *, verbose=False):
    """Fast principal-generator extraction for h=1 number fields (here K=Q(ζ_9)).

    Direct PARI bnf.bnfisprincipal call, skipping principal_generator_compat's
    multi-cascade Sage wrapper. For h(K)=1 the class vector is always trivial
    so the call always succeeds; we still verify class triviality defensively
    and fall back to the compat path if anything looks wrong.

    Accepts either a Sage ideal (legacy) or a PARI HNF + K (fast path used by
    audit B's HNF candidate-ideal builder).

    Returns the same dict shape as principal_generator_compat for drop-in use.
    """
    if K is None:
        # Legacy path: take a Sage ideal.
        I = I_or_hnf
        K = I.number_field()
        compat_arg = I
        try:
            I_hnf = I.pari_hnf()
        except Exception as e:
            if verbose:
                print(f"pari_hnf failed: {e}; falling back to compat")
            return principal_generator_compat(I, proof=False, verbose=verbose)
    else:
        # New path: already a PARI HNF.
        I_hnf = I_or_hnf
        compat_arg = None
    try:
        bnf = _get_pari_bnf(K)
        res = bnf.bnfisprincipal(I_hnf)
        # res is a 2-element PARI vector [class_log_vec, gen_in_integer_basis].
        class_vec = res[0]
        # For h=1 the class log is [] or a zero vector — verify defensively.
        try:
            n_class = int(class_vec.length())
        except Exception:
            n_class = 0
        for i in range(n_class):
            if int(class_vec[i]) != 0:
                # Should never happen for h(K)=1 — fall back rather than lie.
                if compat_arg is None:
                    return {
                        "is_principal": False, "generator": None,
                        "method": "bnfisprincipal_direct",
                        "details": "non-trivial class (unexpected for h=1)",
                    }
                return principal_generator_compat(compat_arg, proof=False, verbose=verbose)
        gen_basis = res[1]
        gen_alg = bnf.nfbasistoalg(gen_basis)
        gen = K(gen_alg)
        return {
            "is_principal": True,
            "generator": gen,
            "method": "bnfisprincipal_direct",
            "details": "h=1 fast path",
        }
    except Exception as e:
        if verbose:
            print(f"bnfisprincipal_direct failed: {e}; falling back to compat")
        if compat_arg is None:
            return {
                "is_principal": False, "generator": None,
                "method": "bnfisprincipal_direct",
                "details": f"failed (HNF input, no compat fallback): {e}",
            }
        return principal_generator_compat(compat_arg, proof=False, verbose=verbose)


def principal_ideal_factorization(F, M):
    """Factor (M)·O_F as a list of (prime_ideal, exponent) pairs.

    Uses PARI's nf.idealfactor() directly instead of Sage's I.factor()
    wrapper — ~2× speedup. Validated to produce identical factorizations
    to the Sage path; falls back to the Sage path on any error.
    """
    if M == 0:
        return []
    try:
        nf = _get_pari_nf(F)
        fac_mat = nf.idealfactor(pari(M))
        n = int(fac_mat.matsize()[0])
        out = []
        for i in range(n):
            prime_pari = fac_mat[i, 0]
            exp = int(fac_mat[i, 1])
            p_rat = ZZ(prime_pari[0])
            gen2_alg = nf.nfbasistoalg(prime_pari[1])
            gen2_F = F(gen2_alg)
            P = F.ideal([p_rat, gen2_F])
            out.append((P, exp))
        return out
    except Exception:
        # Fallback to Sage's path on any PARI error or environment issue.
        return factor_ideal_compat(F.ideal(M))


def extend_prime_to_K(P, K, emb):
    return K.ideal([emb(g) for g in P.gens()])


# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------

def real_embeddings_of_M(F, M, prec=80):
    rr = RealField(prec)
    vals = [rr(s(M)) for s in F.embeddings(rr)]
    vals.sort()
    return vals


def classify_prime_in_extension(P, K, emb):
    PK = extend_prime_to_K(P, K, emb)
    fac = factor_ideal_compat(PK)
    if len(fac) == 1:
        Q, e = fac[0]
        if int(e) == 2:
            return "ramified", fac
        if int(e) == 1:
            return "inert", fac
        return f"unexpected_single_factor_e={e}", fac
    if len(fac) == 2 and int(fac[0][1]) == 1 and int(fac[1][1]) == 1:
        return "split", fac
    return f"unexpected_factorization_len_{len(fac)}", fac



_CLASSIFY_EXTENSION_CACHE = BoundedDict()


def _elt_key(g):
    coeffs = g.polynomial().list()
    return tuple(int(c) for c in coeffs)


def _prime_cache_key(P):
    return tuple(_elt_key(g) for g in P.gens())


def classify_prime_in_extension_from_norm(P):
    """
    Fast screen-only classification for prime ideals P of F in the quadratic
    extension K/F, using only the residue field size q = N(P).

    For F = Q(zeta_9 + zeta_9^{-1}) and K = Q(zeta_9):
      * q divisible by 3  -> ramified
      * q mod 9 in {1,4,7} -> split
      * q mod 9 in {2,5,8} -> inert

    This is sufficient for the inert-odd screen but does not provide ext_fac.
    """
    q = int(P.norm())
    if q % 3 == 0:
        return "ramified"
    r = q % 9
    if r in (1, 4, 7):
        return "split"
    if r in (2, 5, 8):
        return "inert"
    raise RuntimeError(f"unexpected prime ideal norm {q} for screen-only classification")


def classify_prime_in_extension_cached(P, K, emb, cache_profile=None):
    if cache_profile is not None:
        cache_profile["calls"] += 1
        t0 = time.perf_counter()
        key = _prime_cache_key(P)
        cache_profile["t_key"] += time.perf_counter() - t0

        t1 = time.perf_counter()
        if key in _CLASSIFY_EXTENSION_CACHE:
            cache_profile["hits"] += 1
            out = _CLASSIFY_EXTENSION_CACHE[key]
            cache_profile["t_hit_lookup"] += time.perf_counter() - t1
            return out
        cache_profile["misses"] += 1
        t2 = time.perf_counter()
        out = classify_prime_in_extension(P, K, emb)
        _CLASSIFY_EXTENSION_CACHE[key] = out
        cache_profile["t_miss_work"] += time.perf_counter() - t2
        return out

    key = _prime_cache_key(P)
    out = _CLASSIFY_EXTENSION_CACHE.get(key)
    if out is not None:
        return out
    out = classify_prime_in_extension(P, K, emb)
    _CLASSIFY_EXTENSION_CACHE[key] = out
    return out
    

def B2_from_m012(m0, m1, m2):
    return int(2 * int(m0) + 4 * int(m2))


def p3_prime_data(field_data=None):
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data
    pi3 = OF(2 - alpha)
    p3 = F.ideal(pi3)
    return {
        "pi3": pi3,
        "p3": p3,
        "field_data": field_data,
    }


def p3_valuation_of_M(M, field_data=None):
    pdata = p3_prime_data(field_data=field_data)
    p3 = pdata["p3"]
    if M == 0:
        return None
    try:
        return int(FactorialIdealCompatWrapper := p3.valuation(FactorialIdealCompatWrapper))  # dummy to force except path off
    except Exception:
        pass
    try:
        return int(FactorialIdealCompatWrapper := pdata["field_data"][0].ideal(M).valuation(p3))
    except Exception:
        # fallback by repeated divisibility
        pi3 = pdata["pi3"]
        x = pdata["field_data"][0](M)
        e = 0
        while True:
            try:
                q = x / (pi3 ** (e + 1))
                if q in pdata["field_data"][1]:
                    e += 1
                else:
                    break
            except Exception:
                break
        return e


def unit_part_mod_p3_is_one(M, field_data=None):
    """
    Cheap necessary local test at p3:
    after removing the p3-adic valuation, the remaining unit must be 1 mod p3.
    """
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data
    if M == 0:
        return True

    pdata = p3_prime_data(field_data=field_data)
    pi3 = pdata["pi3"]
    p3 = pdata["p3"]

    try:
        e = int(F.ideal(M).valuation(p3))
    except Exception:
        e = 0
        x = F(M)
        while True:
            try:
                q = x / (pi3 ** (e + 1))
                if q in OF:
                    e += 1
                else:
                    break
            except Exception:
                break

    try:
        u = F(M) / (pi3 ** e)
    except Exception:
        return False

    # Need the unit part to be integral before reducing mod p3.
    if u not in OF:
        return False

    try:
        return (u - 1) in p3
    except Exception:
        try:
            return F.ideal(u - 1).valuation(p3) >= 1
        except Exception:
            return False


def quick_local_p3_norm_screen(m0, m1, m2, field_data=None):
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data
    M = F(ZZ(m0) + ZZ(m1) * alpha + ZZ(m2) * alpha**2)

    if M == 0:
        return {
            "status": "ZERO",
            "passes": True,
        }

    ok = unit_part_mod_p3_is_one(M, field_data=field_data)
    return {
        "status": "PASSES_LOCAL_P3" if ok else "REJECT_LOCAL_P3",
        "passes": bool(ok),
    }
    

# ----------------------------------------------------------------------
# Main analysis / screening
# ----------------------------------------------------------------------

def analyze_M(coeffs, prec=80, field_data=None):
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data

    m0, m1, m2 = [ZZ(c) for c in coeffs]
    M = F(m0 + m1 * alpha + m2 * alpha**2)

    if m0 == 0 and m1 == 0 and m2 == 0:
        return {
            "input": (m0, m1, m2),
            "M": M,
            "real_embeddings": [ZZ(0), ZZ(0), ZZ(0)],
            "all_real_nonnegative": True,
            "factorization_F": [],
            "prime_data": [],
            "inert_odd_obstruction": False,
            "status": "ZERO",
            "field_data": field_data,
            "B2": 0,
            "B": 0,
            "split_prime_count": 0,
            "inert_prime_count": 0,
            "ramified_prime_count": 0,
        }

    real_vals = real_embeddings_of_M(F, M, prec=prec)
    fac = principal_ideal_factorization(F, M)

    prime_data = []
    inert_odd = False
    for P, e in fac:
        e_int = int(e)
        kind, ext_fac = classify_prime_in_extension_cached(P, K, emb)
        item = {
            "P": P,
            "e": e_int,
            "P_norm": ZZ(P.norm()),
            "kind": kind,
            "ext_fac": ext_fac,
        }
        prime_data.append(item)
        if kind == "inert" and (e_int % 2 == 1):
            inert_odd = True

    split_count = sum(1 for x in prime_data if x["kind"] == "split")
    inert_count = sum(1 for x in prime_data if x["kind"] == "inert")
    ram_count = sum(1 for x in prime_data if x["kind"] == "ramified")
    status = "REJECT_INERT_ODD" if inert_odd else "PASSES_IDEAL_SIEVE"
    B2 = B2_from_m012(m0, m1, m2)
    return {
        "input": (m0, m1, m2),
        "M": M,
        "real_embeddings": real_vals,
        "all_real_nonnegative": all(v >= 0 for v in real_vals),
        "factorization_F": fac,
        "prime_data": prime_data,
        "inert_odd_obstruction": inert_odd,
        "status": status,
        "field_data": field_data,
        "B2": B2,
        "B": int(math.isqrt(B2)) if B2 >= 0 else None,
        "split_prime_count": split_count,
        "inert_prime_count": inert_count,
        "ramified_prime_count": ram_count,
    }


def quick_screen_M(
    m0, m1, m2,
    check_real_embeddings=False,
    prec=80,
    field_data=None,
    check_local_p3k=False,
    local_p3k=2,
):
    analysis = analyze_M((m0, m1, m2), prec=prec, field_data=field_data)

    if analysis["status"] == "ZERO":
        status = "ZERO"
    elif check_real_embeddings and not analysis["all_real_nonnegative"]:
        status = "REJECT_NEGATIVE_EMBEDDING"
    else:
        status = analysis["status"]

    if status == "PASSES_IDEAL_SIEVE" and check_local_p3k:
        p3scr = quick_local_p3k_norm_screen(
            m0, m1, m2,
            field_data=analysis["field_data"],
            k=local_p3k,
        )
        if not p3scr["passes"]:
            status = "REJECT_LOCAL_P3K"

    return {
        "input": analysis["input"],
        "status": status,
        "split_prime_count": analysis["split_prime_count"],
        "inert_prime_count": analysis["inert_prime_count"],
        "ramified_prime_count": analysis["ramified_prime_count"],
        "all_real_nonnegative": analysis["all_real_nonnegative"],
        "B2": analysis["B2"],
        "B": analysis["B"],
    }
    



def analyze_M_fastscreen(coeffs, prec=80, field_data=None):
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data

    m0, m1, m2 = [ZZ(c) for c in coeffs]
    M = F(m0 + m1 * alpha + m2 * alpha**2)

    if m0 == 0 and m1 == 0 and m2 == 0:
        return {
            "input": (m0, m1, m2),
            "M": M,
            "real_embeddings": [ZZ(0), ZZ(0), ZZ(0)],
            "all_real_nonnegative": True,
            "factorization_F": [],
            "inert_odd_obstruction": False,
            "status": "ZERO",
            "field_data": field_data,
            "B2": 0,
            "B": 0,
            "split_prime_count": 0,
            "inert_prime_count": 0,
            "ramified_prime_count": 0,
        }

    real_vals = real_embeddings_of_M(F, M, prec=prec)
    fac = principal_ideal_factorization(F, M)

    inert_odd = False
    split_count = 0
    inert_count = 0
    ram_count = 0
    for P, e in fac:
        e_int = int(e)
        kind = classify_prime_in_extension_from_norm(P)
        if kind == "split":
            split_count += 1
        elif kind == "inert":
            inert_count += 1
            if (e_int % 2) == 1:
                inert_odd = True
        elif kind == "ramified":
            ram_count += 1
        else:
            raise RuntimeError(f"unexpected fast screen classification: {kind}")

    status = "REJECT_INERT_ODD" if inert_odd else "PASSES_IDEAL_SIEVE"
    B2 = B2_from_m012(m0, m1, m2)
    return {
        "input": (m0, m1, m2),
        "M": M,
        "real_embeddings": real_vals,
        "all_real_nonnegative": all(v >= 0 for v in real_vals),
        "factorization_F": fac,
        "inert_odd_obstruction": inert_odd,
        "status": status,
        "field_data": field_data,
        "B2": B2,
        "B": int(math.isqrt(B2)) if B2 >= 0 else None,
        "split_prime_count": split_count,
        "inert_prime_count": inert_count,
        "ramified_prime_count": ram_count,
    }


def quick_screen_M_fast(
    m0, m1, m2,
    check_real_embeddings=False,
    prec=80,
    field_data=None,
    check_local_p3k=False,
    local_p3k=2,
):
    analysis = analyze_M_fastscreen((m0, m1, m2), prec=prec, field_data=field_data)

    if analysis["status"] == "ZERO":
        status = "ZERO"
    elif check_real_embeddings and not analysis["all_real_nonnegative"]:
        status = "REJECT_NEGATIVE_EMBEDDING"
    else:
        status = analysis["status"]

    if status == "PASSES_IDEAL_SIEVE" and check_local_p3k:
        p3scr = quick_local_p3k_norm_screen(
            m0, m1, m2,
            field_data=analysis["field_data"],
            k=local_p3k,
        )
        if not p3scr["passes"]:
            status = "REJECT_LOCAL_P3K"

    return {
        "input": analysis["input"],
        "status": status,
        "split_prime_count": analysis["split_prime_count"],
        "inert_prime_count": analysis["inert_prime_count"],
        "ramified_prime_count": analysis["ramified_prime_count"],
        "all_real_nonnegative": analysis["all_real_nonnegative"],
        "B2": analysis["B2"],
        "B": analysis["B"],
    }
def _k_elem_to_coeff6(x, K):
    try:
        coeffs = K(x).polynomial().list()
    except Exception:
        try:
            coeffs = list(K(x))
        except Exception:
            return None
    coeffs = [ZZ(c) for c in coeffs]
    if len(coeffs) > 6:
        return None
    coeffs += [ZZ(0)] * (6 - len(coeffs))
    return tuple(coeffs)


def lift_K_to_F_if_possible(x, field_data=None):
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data
    try:
        xK = K(x)
    except Exception:
        return None
    try:
        if xK != xK.conjugate():
            return None
    except Exception:
        return None
    c = _k_elem_to_coeff6(xK, K)
    if c is None:
        return None
    c0,c1,c2,c3,c4,c5 = c
    if c3 != 0:
        return None
    if c1 + c2 != 0:
        return None
    if c1 - c4 + c5 != 0:
        return None
    m2 = -c4
    m1 = -c5
    m0 = c0 - 2*m2
    try:
        return F(m0 + m1*alpha + m2*alpha**2)
    except Exception:
        return None


class UnitNormSolver:
    """
    Cached exact solver for the unit norm map N_{K/F}: O_K^* -> O_F^*.

    Works on the free parts of the unit groups via exponent vectors, then
    verifies the result exactly in the field.

    Methods
    -------
    solve(u):
        Return a dictionary:
            {
                "success": bool,
                "epsilon": eps or None,
                "reason": str or None,
                "target_log": ...,
                "solution_vector": ...,
            }
    """

    def __init__(self, field_data=None):
        if field_data is None:
            field_data = build_fields()
        self.field_data = field_data
        self.F, self.OF, self.K, self.OK, self.alpha, self.z, self.emb = field_data

        self._built = False
        self._build_error = None

        self.UF = None
        self.UK = None

        self.uf_gens = None
        self.uk_gens = None

        self.norm_images = None
        self.norm_matrix = None

        self.S = None
        self.U = None
        self.V = None

        self._build()

    def _build(self):
        try:
            self.UF = self.F.unit_group()
            self.UK = self.K.unit_group()

            self.uf_gens = list(self.UF.gens_values())
            self.uk_gens = list(self.UK.gens_values())

            cols = []
            norm_images = []

            for kg in self.uk_gens:
                nkg = lift_K_to_F_if_possible(kg * kg.conjugate(), self.field_data)
                if nkg is None:
                    self._build_error = "FAILED_TO_LIFT_NORM_OF_K_UNIT"
                    return

                norm_images.append(nkg)

                try:
                    logv = self.UF.log(nkg)
                except Exception as e:
                    self._build_error = f"UF_LOG_FAILED_ON_NORM_IMAGE: {e}"
                    return

                cols.append(vector(ZZ, [ZZ(v) for v in logv]))

            self.norm_images = norm_images

            r = len(self.UF.gens())
            s = len(self.uk_gens)

            A = matrix(ZZ, r, s)
            for j, col in enumerate(cols):
                for i, val in enumerate(col):
                    A[i, j] = ZZ(val)

            self.norm_matrix = A

            self.S, self.U, self.V = A.smith_form()
            self._built = True

        except Exception as e:
            self._build_error = f"UNIT_SOLVER_BUILD_FAILED: {e}"
            self._built = False

    def _solve_integer_system(self, b):
        """
        Solve A x = b over Z using Smith normal form:
            U A V = S
        """
        if not self._built:
            return {
                "success": False,
                "reason": self._build_error or "UNIT_SOLVER_NOT_BUILT",
                "solution_vector": None,
            }

        try:
            rhs = self.U * b
        except Exception as e:
            return {
                "success": False,
                "reason": f"SMITH_TRANSFORM_FAILED: {e}",
                "solution_vector": None,
            }

        y = [ZZ(0)] * self.norm_matrix.ncols()
        rank = min(self.S.nrows(), self.S.ncols())

        for i in range(rank):
            d = self.S[i, i]
            if d == 0:
                if rhs[i] != 0:
                    return {
                        "success": False,
                        "reason": f"SMITH_ZERO_ROW_OBSTRUCTION_AT_{i}",
                        "solution_vector": None,
                    }
            else:
                if rhs[i] % d != 0:
                    return {
                        "success": False,
                        "reason": f"SMITH_DIVISIBILITY_OBSTRUCTION_AT_{i}",
                        "solution_vector": None,
                    }
                y[i] = rhs[i] // d

        for i in range(rank, len(rhs)):
            if rhs[i] != 0:
                return {
                    "success": False,
                    "reason": f"SMITH_TAIL_OBSTRUCTION_AT_{i}",
                    "solution_vector": None,
                }

        try:
            xvec = self.V * vector(ZZ, y)
        except Exception as e:
            return {
                "success": False,
                "reason": f"SMITH_BACKSUBSTITUTION_FAILED: {e}",
                "solution_vector": None,
            }

        return {
            "success": True,
            "reason": None,
            "solution_vector": xvec,
        }

    def solve(self, u):
        """
        Try to find epsilon in O_K^* such that epsilon*conjugate(epsilon) = u.

        Returns
        -------
        dict
            {
                "success": bool,
                "epsilon": eps or None,
                "reason": str or None,
                "target_log": tuple or None,
                "solution_vector": tuple or None,
            }
        """
        if not self._built:
            return {
                "success": False,
                "epsilon": None,
                "reason": self._build_error or "UNIT_SOLVER_NOT_BUILT",
                "target_log": None,
                "solution_vector": None,
            }

        try:
            uF = self.F(u)
        except Exception as e:
            return {
                "success": False,
                "epsilon": None,
                "reason": f"TARGET_NOT_IN_F: {e}",
                "target_log": None,
                "solution_vector": None,
            }

        try:
            if not uF.is_unit():
                return {
                    "success": False,
                    "epsilon": None,
                    "reason": "TARGET_NOT_A_UNIT",
                    "target_log": None,
                    "solution_vector": None,
                }
        except Exception:
            pass

        try:
            blog = vector(ZZ, [ZZ(v) for v in self.UF.log(uF)])
        except Exception as e:
            return {
                "success": False,
                "epsilon": None,
                "reason": f"UF_LOG_TARGET_FAILED: {e}",
                "target_log": None,
                "solution_vector": None,
            }

        lin = self._solve_integer_system(blog)
        if not lin["success"]:
            return {
                "success": False,
                "epsilon": None,
                "reason": lin["reason"],
                "target_log": tuple(int(v) for v in blog),
                "solution_vector": None,
            }

        xvec = lin["solution_vector"]

        try:
            eps = self.K(1)
            for kg, e in zip(self.uk_gens, xvec):
                ee = int(e)
                if ee:
                    eps *= kg**ee
        except Exception as e:
            return {
                "success": False,
                "epsilon": None,
                "reason": f"UNIT_RECONSTRUCTION_FAILED: {e}",
                "target_log": tuple(int(v) for v in blog),
                "solution_vector": tuple(int(v) for v in xvec),
            }

        try:
            lifted = lift_K_to_F_if_possible(eps * eps.conjugate(), self.field_data)
        except Exception as e:
            return {
                "success": False,
                "epsilon": None,
                "reason": f"NORM_VERIFICATION_FAILED: {e}",
                "target_log": tuple(int(v) for v in blog),
                "solution_vector": tuple(int(v) for v in xvec),
            }

        if lifted is None:
            return {
                "success": False,
                "epsilon": None,
                "reason": "NORM_VERIFICATION_LIFT_FAILED",
                "target_log": tuple(int(v) for v in blog),
                "solution_vector": tuple(int(v) for v in xvec),
            }

        if lifted != uF:
            return {
                "success": False,
                "epsilon": None,
                "reason": "NORM_VERIFICATION_MISMATCH",
                "target_log": tuple(int(v) for v in blog),
                "solution_vector": tuple(int(v) for v in xvec),
            }

        return {
            "success": True,
            "epsilon": eps,
            "reason": None,
            "target_log": tuple(int(v) for v in blog),
            "solution_vector": tuple(int(v) for v in xvec),
        }
        

_UNIT_NORM_SOLVER_CACHE = BoundedDict()

def get_unit_norm_solver(field_data=None):
    if field_data is None:
        field_data = build_fields()
    key = tuple(id(x) for x in field_data)
    solver = _UNIT_NORM_SOLVER_CACHE.get(key)
    if solver is None:
        solver = UnitNormSolver(field_data=field_data)
        _UNIT_NORM_SOLVER_CACHE[key] = solver
    return solver


def unit_norm_preimage_in_K(u, field_data=None):
    solver = get_unit_norm_solver(field_data=field_data)
    res = solver.solve(u)
    return res["epsilon"] if res["success"] else None


def unit_is_global_norm_in_K_over_F(u, field_data=None):
    solver = get_unit_norm_solver(field_data=field_data)
    return solver.solve(u)["success"]


# ----------------------------------------------------------------------
# Local p3-adic norm screen modulo p3^k
# ----------------------------------------------------------------------

_LOCAL_P3_IMAGE_CACHE = BoundedDict()

def p3_data(field_data=None):
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data
    piF = OF(2 - alpha)     # prime above 3 in F
    piK = OK(1 - z)         # prime above 3 in K, N(piK)=piF
    p3F = F.ideal(piF)
    p3K = K.ideal(piK)
    return {
        "field_data": field_data,
        "piF": piF,
        "piK": piK,
        "p3F": p3F,
        "p3K": p3K,
    }


def valuation_at_p3_F(x, field_data=None):
    """
    p3-adic valuation in F.
    Returns None for x = 0.
    """
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data
    if x == 0:
        return None

    pdata = p3_data(field_data=field_data)
    p3F = pdata["p3F"]

    try:
        return int(F.ideal(x).valuation(p3F))
    except Exception:
        # fallback by repeated divisibility with piF
        piF = pdata["piF"]
        xx = F(x)
        e = 0
        while True:
            try:
                q = xx / (piF ** (e + 1))
                if q in OF:
                    e += 1
                else:
                    break
            except Exception:
                break
        return e


def unit_part_at_p3_F(x, field_data=None):
    """
    Write x = piF^e * u with u a p3-unit in O_F, and return (e,u).
    For x=0 returns (None,None).
    """
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data

    if x == 0:
        return None, None

    pdata = p3_data(field_data=field_data)
    piF = pdata["piF"]

    e = valuation_at_p3_F(x, field_data=field_data)
    u = F(x) / (piF ** e)
    return e, u


def _unit_reps_mod_pi_power_F(k, field_data=None):
    """
    Representatives of unit classes in O_F / (piF^k),
    using the fact that the residue field is F_3.
    """
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data
    pdata = p3_data(field_data=field_data)
    piF = pdata["piF"]

    reps = []
    digits = [0, 1, 2]
    for a0 in (1, 2):
        if k == 1:
            coeff_lists = [()]
        else:
            coeff_lists = product(digits, repeat=k - 1)
        for tail in coeff_lists:
            x = F(a0)
            for i, ai in enumerate(tail, start=1):
                x += F(ai) * (piF ** i)
            reps.append(OF(x))
    return reps


def _unit_reps_mod_pi_power_K(k, field_data=None):
    """
    Representatives of unit classes in O_K / (piK^k),
    again using residue field F_3.
    """
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data
    pdata = p3_data(field_data=field_data)
    piK = pdata["piK"]

    reps = []
    digits = [0, 1, 2]
    for a0 in (1, 2):
        if k == 1:
            coeff_lists = [()]
        else:
            coeff_lists = product(digits, repeat=k - 1)
        for tail in coeff_lists:
            x = K(a0)
            for i, ai in enumerate(tail, start=1):
                x += K(ai) * (piK ** i)
            reps.append(OK(x))
    return reps


def _congruent_mod_p3_power_F(x, y, k, field_data=None):
    """
    Check x ≡ y mod p3^k in O_F.
    """
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data
    pdata = p3_data(field_data=field_data)
    p3F = pdata["p3F"]

    try:
        return F.ideal(F(x) - F(y)).valuation(p3F) >= k
    except Exception:
        try:
            return (F(x) - F(y)) in (p3F ** k)
        except Exception:
            return False


def local_norm_image_mod_p3k(k=2, field_data=None):
    """
    Precompute the image of local norms on unit classes modulo p3^k:
        N((O_K / piK^k)^*) subset (O_F / piF^k)^*.
    Returned as a list of chosen representatives in O_F.
    """
    if field_data is None:
        field_data = build_fields()

    key = (k, tuple(id(x) for x in field_data))
    cached = _LOCAL_P3_IMAGE_CACHE.get(key)
    if cached is not None:
        return cached

    F, OF, K, OK, alpha, z, emb = field_data
    repsF = _unit_reps_mod_pi_power_F(k, field_data=field_data)
    repsK = _unit_reps_mod_pi_power_K(2 * k, field_data=field_data)

    image = []
    seen = set()

    for x in repsK:
        n = lift_K_to_F_if_possible(x * x.conjugate(), field_data=field_data)
        if n is None:
            continue

        # identify the class by matching to one of the chosen F-representatives
        for y in repsF:
            if _congruent_mod_p3_power_F(n, y, k, field_data=field_data):
                keyy = tuple(_k_elem_to_coeff6(K(emb(y)), K))  # stable hashable key
                if keyy not in seen:
                    seen.add(keyy)
                    image.append(y)
                break

    _LOCAL_P3_IMAGE_CACHE[key] = image
    return image


def quick_local_p3k_norm_screen(m0, m1, m2, field_data=None, k=2):
    """
    Necessary local norm test at p3^k:
    strip the p3-valuation of M and check whether the unit part modulo p3^k
    lies in the precomputed norm image.
    """
    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data

    M = F(ZZ(m0) + ZZ(m1) * alpha + ZZ(m2) * alpha**2)
    if M == 0:
        return {"status": "ZERO", "passes": True}

    e, u = unit_part_at_p3_F(M, field_data=field_data)
    if u is None:
        return {"status": "REJECT_LOCAL_P3K", "passes": False}

    if u not in OF:
        return {"status": "REJECT_LOCAL_P3K", "passes": False}

    image = local_norm_image_mod_p3k(k=k, field_data=field_data)
    ok = any(_congruent_mod_p3_power_F(u, y, k, field_data=field_data) for y in image)

    return {
        "status": "PASSES_LOCAL_P3K" if ok else "REJECT_LOCAL_P3K",
        "passes": bool(ok),
    }
    
    
# ----------------------------------------------------------------------
# Candidate ideal enumeration
# ----------------------------------------------------------------------

def _build_candidate_ideal_data(analysis):
    choices = []
    for item in analysis["prime_data"]:
        P, e, kind, ext_fac = item["P"], item["e"], item["kind"], item["ext_fac"]
        if kind == "split":
            (Q1, e1), (Q2, e2) = ext_fac
            local = []
            for u in range(e + 1):
                local.append({"kind": "split", "P": P, "e": e, "Q1": Q1, "Q2": Q2, "u": u})
            choices.append(local)
        elif kind == "inert":
            Q, e_up = ext_fac[0]
            if e % 2 != 0:
                return None
            choices.append([{"kind": "inert", "P": P, "e": e, "Q": Q, "u": e // 2}])
        elif kind == "ramified":
            Q, e_up = ext_fac[0]
            choices.append([{"kind": "ramified", "P": P, "e": e, "Q": Q, "u": e}])
        else:
            raise RuntimeError(f"Unexpected prime behavior: {kind}")
    return choices


_IDEAL_PARI_HNF_CACHE = BoundedDict()


def _get_pari_hnf_of_ideal(I):
    """Cached PARI HNF for a Sage ideal. Cache by Python id (safe since prime
    ideals are reused across many candidate enumerations per stage-3 run)."""
    key = id(I)
    h = _IDEAL_PARI_HNF_CACHE.get(key)
    if h is None:
        h = I.pari_hnf()
        _IDEAL_PARI_HNF_CACHE[key] = h
    return h


def _build_ideal_hnf_from_choice(choice_tuple, K):
    """PARI-direct candidate-ideal construction.

    Replaces Sage's K.ideal(1) * Q1^u * Q2^(e-u) cascade (53.8% of stage-3
    wall per audit B) with PARI's idealpow + idealmul on cached prime HNFs.
    Returns (pari_hnf, summary) — the HNF is passed straight to
    principal_generator_h1_fast (no Sage ideal materialization)."""
    bnf = _get_pari_bnf(K)
    summary = []
    hnf = None
    for c in choice_tuple:
        if c["kind"] == "split":
            Q1, Q2, u, e = c["Q1"], c["Q2"], c["u"], c["e"]
            h1 = bnf.idealpow(_get_pari_hnf_of_ideal(Q1), u)
            h2 = bnf.idealpow(_get_pari_hnf_of_ideal(Q2), e - u)
            mult = bnf.idealmul(h1, h2)
            hnf = mult if hnf is None else bnf.idealmul(hnf, mult)
            summary.append(("split", u, e - u, str(Q1), str(Q2)))
        elif c["kind"] == "inert":
            Q, u = c["Q"], c["u"]
            h = bnf.idealpow(_get_pari_hnf_of_ideal(Q), u)
            hnf = h if hnf is None else bnf.idealmul(hnf, h)
            summary.append(("inert", u, str(Q)))
        else:  # ramified
            Q, u = c["Q"], c["u"]
            h = bnf.idealpow(_get_pari_hnf_of_ideal(Q), u)
            hnf = h if hnf is None else bnf.idealmul(hnf, h)
            summary.append(("ramified", u, str(Q)))
    if hnf is None:
        # No primes (empty product) → unit ideal as PARI matid.
        hnf = pari.matid(K.degree())
    return hnf, summary


def _build_ideal_from_choice(choice_tuple, K):
    """Sage-ideal version (slow, only used by diagnostics). Kept for
    backward compat with process_M and roots_profiled."""
    I = K.ideal(1)
    summary = []
    for c in choice_tuple:
        if c["kind"] == "split":
            Q1, Q2, u, e = c["Q1"], c["Q2"], c["u"], c["e"]
            I *= (Q1 ** u) * (Q2 ** (e - u))
            summary.append(("split", u, e - u, str(Q1), str(Q2)))
        elif c["kind"] == "inert":
            Q, u = c["Q"], c["u"]
            I *= Q ** u
            summary.append(("inert", u, str(Q)))
        else:
            Q, u = c["Q"], c["u"]
            I *= Q ** u
            summary.append(("ramified", u, str(Q)))
    return I, summary


def enumerate_candidate_ideals(coeffs_or_analysis, max_candidates=None, prec=80, field_data=None):
    if isinstance(coeffs_or_analysis, dict):
        analysis = coeffs_or_analysis
        field_data = analysis["field_data"]
    else:
        analysis = analyze_M(coeffs_or_analysis, prec=prec, field_data=field_data)
        field_data = analysis["field_data"]
    F, OF, K, OK, alpha, z, emb = field_data
    if analysis["status"] == "ZERO":
        return []
    if analysis["inert_odd_obstruction"]:
        return []
    choice_lists = _build_candidate_ideal_data(analysis)
    if choice_lists is None:
        return []
    out = []
    for idx, choice_tuple in enumerate(product(*choice_lists)):
        # Audit B: PARI HNF construction (replaces slow Sage K.ideal mult).
        hnf, summary = _build_ideal_hnf_from_choice(choice_tuple, K)
        out.append({"ideal_hnf": hnf, "local_choice_summary": summary, "K": K})
        if max_candidates is not None and idx + 1 >= max_candidates:
            break
    return out


def principal_generator_compat(I, *, proof=False, verbose=False):
    """
    Try hard to extract a generator g for a principal number-field ideal I.

    Returns a dictionary:
        {
            "is_principal": bool,
            "generator": g or None,
            "method": str or None,
            "details": str or None,
        }

    The function tries several Sage APIs and verifies any candidate by checking
    that the principal ideal generated by g equals I.
    """
    K = I.number_field()
    zero = K(0)

    def _verify(gen, method):
        if gen is None:
            return None

        try:
            if gen == zero:
                return None
        except Exception:
            pass

        try:
            J = K.ideal(gen)
        except Exception as e:
            return {
                "is_principal": False,
                "generator": None,
                "method": method,
                "details": f"candidate ideal construction failed: {e}",
            }

        try:
            if J == I:
                return {
                    "is_principal": True,
                    "generator": gen,
                    "method": method,
                    "details": "verified by ideal equality",
                }
        except Exception:
            pass

        try:
            same_norm = (J.absolute_norm() == I.absolute_norm())
        except Exception:
            try:
                same_norm = (J.norm() == I.norm())
            except Exception:
                same_norm = False

        try:
            same_containment = J.is_subset(I) and I.is_subset(J)
        except Exception:
            same_containment = False

        if same_norm and same_containment:
            return {
                "is_principal": True,
                "generator": gen,
                "method": method,
                "details": "verified by mutual containment + norm",
            }

        return {
            "is_principal": False,
            "generator": None,
            "method": method,
            "details": "candidate did not verify as generator",
        }

    try:
        if I == K.ideal(1):
            return {
                "is_principal": True,
                "generator": K(1),
                "method": "unit_ideal",
                "details": "I = (1)",
            }
    except Exception:
        pass

    # 1. Try is_principal(proof=...) returning (bool, gen)
    try:
        out = I.is_principal(proof=proof)
        if isinstance(out, tuple) and len(out) == 2:
            is_princ, gen = out
            if is_princ:
                res = _verify(gen, "is_principal_tuple")
                if res and res["is_principal"]:
                    return res
            else:
                return {
                    "is_principal": False,
                    "generator": None,
                    "method": "is_principal_tuple",
                    "details": "reported non-principal",
                }
    except TypeError:
        pass
    except Exception as e:
        last1 = f"is_principal(proof=...) failed: {e}" if verbose else None
    else:
        last1 = None

    # 2. Try is_principal() returning (bool, gen) or bool
    try:
        out = I.is_principal()
        if isinstance(out, tuple) and len(out) == 2:
            is_princ, gen = out
            if is_princ:
                res = _verify(gen, "is_principal_noarg_tuple")
                if res and res["is_principal"]:
                    return res
            else:
                return {
                    "is_principal": False,
                    "generator": None,
                    "method": "is_principal_noarg_tuple",
                    "details": "reported non-principal",
                }
        elif isinstance(out, (bool, ZZ.__class__, int)):
            if not bool(out):
                return {
                    "is_principal": False,
                    "generator": None,
                    "method": "is_principal_bool",
                    "details": "reported non-principal",
                }
    except Exception as e:
        last2 = f"is_principal() failed: {e}" if verbose else None
    else:
        last2 = None

    # 3. Try gens_reduced()
    try:
        gens_red = I.gens_reduced()
        if gens_red is not None:
            for g in gens_red:
                res = _verify(g, "gens_reduced")
                if res and res["is_principal"]:
                    return res
    except Exception as e:
        last3 = f"gens_reduced() failed: {e}" if verbose else None
    else:
        last3 = None

    # 4. Try gens()
    try:
        gens = I.gens()
        if gens is not None:
            for g in gens:
                res = _verify(g, "gens")
                if res and res["is_principal"]:
                    return res
    except Exception as e:
        last4 = f"gens() failed: {e}" if verbose else None
    else:
        last4 = None

    # 5. Small linear combinations of generators
    try:
        gens = [g for g in I.gens() if g != zero]
        coeffs = [-2, -1, 0, 1, 2]
        seen = set()

        for tup in product(coeffs, repeat=len(gens)):
            if all(c == 0 for c in tup):
                continue

            try:
                g = sum(K(c) * gens[i] for i, c in enumerate(tup))
            except Exception:
                continue

            try:
                key = tuple(g.polynomial().coefficients(sparse=False))
            except Exception:
                key = None

            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)

            res = _verify(g, "small_linear_combination")
            if res and res["is_principal"]:
                return res
    except Exception as e:
        last5 = f"small linear combination search failed: {e}" if verbose else None
    else:
        last5 = None

    details = "; ".join(
        s for s in (last1, last2, last3, last4, last5) if s is not None
    ) or "no verified generator found"

    return {
        "is_principal": False,
        "generator": None,
        "method": None,
        "details": details,
    }

    
# ----------------------------------------------------------------------
# Constructive helpers
# ----------------------------------------------------------------------

def orbit_expand_roots(roots):
    from .tools import mul, DEG

    def _reduce_coeffs(coeffs):
        c = list(coeffs)
        if len(c) < DEG:
            c += [0] * (DEG - len(c))
        while len(c) > DEG:
            k = len(c) - 1
            v = c.pop()
            if v == 0:
                continue
            if k - 3 >= len(c):
                c += [0] * (k - 3 - len(c) + 1)
            if k - 6 >= len(c):
                c += [0] * (k - 6 - len(c) + 1)
            c[k - 3] -= v
            c[k - 6] -= v
        if len(c) < DEG:
            c += [0] * (DEG - len(c))
        return tuple(int(x) for x in c[:DEG])

    monos = []
    for k in range(9):
        coeffs = [0] * (k + 1)
        coeffs[k] = 1
        m = _reduce_coeffs(coeffs)
        monos.append(m)
        monos.append(tuple(-x for x in m))   # add -zeta^k too

    out = set()
    for r in roots:
        rr = tuple(int(x) for x in r)
        for m in monos:
            out.add(tuple(mul(m, rr)))
    return tuple(sorted(out))


def direct_roots_via_legacy_backend(coeffs, verbose=False, rank=None, orbit=True):
    m0, m1, m2 = [int(c) for c in coeffs]

    from .tools import solve_n0_n3_all, verify_b_matches_M012, trace_from_real_coeffs

    trM = trace_from_real_coeffs(m0, m1, m2)
    if trM < 0:
        return []
    B2 = B2_from_m012(m0, m1, m2)
    if B2 < 0:
        return []
    B = int(math.isqrt(B2))
    if verbose:
        prefix = f"rank {rank}: " if rank is not None else ""
        print(f"{prefix}zeta9.roots direct solve Y={(m0,m1,m2)}, B2={B2}, B={B}")
    roots=[]; seen=set()
    for n1 in range(-B, B+1):
        for n2 in range(-B, B+1):
            for n4 in range(-B, B+1):
                for n5 in range(-B, B+1):
                    for n0, n3 in solve_n0_n3_all(n1, n2, n4, n5, m1, m2, B):
                        n=(n0,n1,n2,n3,n4,n5)
                        if sum(x*x for x in n) > B2:
                            continue
                        if verify_b_matches_M012(n, m0, m1, m2) and n not in seen:
                            seen.add(n)
                            roots.append(n)
    if orbit:
        return list(orbit_expand_roots(roots))
    return roots


def _root_from_field_generator(gen, K):
    # Represent generator in power basis 1,z,...,z^5 if integral.
    try:
        coeffs = K(gen).polynomial().list()
    except Exception:
        try:
            coeffs = K(gen).list()
        except Exception:
            return None
    coeffs = [ZZ(c) for c in coeffs]
    if len(coeffs) > 6:
        return None
    coeffs += [0]*(6-len(coeffs))
    return tuple(int(c) for c in coeffs)


def simple_constructive_roots(coeffs, field_data=None, max_candidates=256, verbose=False):
    analysis = analyze_M(coeffs, field_data=field_data)
    F, OF, K, OK, alpha, z, emb = analysis["field_data"]
    m0, m1, m2 = [int(c) for c in coeffs]
    diagnostics = {
        "candidate_count": 0,
        "principal_candidate_count": 0,
        "generator_count": 0,
        "failure_reason": None,
        "generator_method": None,
        "generator_details": None,
        "candidate_summaries": [],
    }
    
    unit_solver = get_unit_norm_solver(field_data=analysis["field_data"])
    diagnostics["unit_solver_built"] = unit_solver._built
    diagnostics["unit_solver_build_error"] = unit_solver._build_error
    
    cands = enumerate_candidate_ideals(analysis, max_candidates=max_candidates)
    diagnostics["candidate_count"] = len(cands)
    roots = []
    seen = set()
    if not cands:
        diagnostics["failure_reason"] = "NO_CANDIDATE_IDEALS"
        return roots, diagnostics

    M = analysis["M"]
    saw_principal_report = False

    for idx, cand in enumerate(cands):
        # PARI HNF + K direct (audit B): no Sage ideal materialization, no
        # .pari_hnf() reconversion. bnfisprincipal is the only real work.
        pg = principal_generator_h1_fast(cand["ideal_hnf"], K=cand["K"], verbose=verbose)

        cand_diag = {
            "index": idx,
            "summary": cand["local_choice_summary"],
            "principal": pg["is_principal"],
            "generator_method": pg.get("method"),
            "generator_details": pg.get("details"),
        }
        diagnostics["candidate_summaries"].append(cand_diag)

        if not pg["is_principal"]:
            cand_diag["reason"] = "NOT_PRINCIPAL_OR_UNVERIFIED"
            continue

        saw_principal_report = True
        diagnostics["principal_candidate_count"] += 1

        gen = pg["generator"]
        if gen is None:
            cand_diag["reason"] = "NO_GENERATOR_RETURNED"
            diagnostics["generator_method"] = pg.get("method")
            diagnostics["generator_details"] = pg.get("details")
            continue

        diagnostics["generator_count"] += 1
        diagnostics["generator_method"] = pg.get("method")
        diagnostics["generator_details"] = pg.get("details")

        try:
            ratio = (gen * gen.conjugate()) / emb(M)
        except Exception as e:
            cand_diag["reason"] = "RATIO_COMPUTE_FAILED"
            cand_diag["ratio_error"] = str(e)
            continue

        try:
            if ratio == 1:
                r = _root_from_field_generator(gen, K)
                if r is not None and r not in seen:
                    seen.add(r)
                    roots.append(r)
                cand_diag["reason"] = "EXACT_RATIO_ONE"
                continue
        except Exception:
            pass

        ratioF = lift_K_to_F_if_possible(ratio, field_data=analysis["field_data"])
        if ratioF is None:
            cand_diag["reason"] = "RATIO_NOT_IN_BASE_FIELD"
            continue

        solver = get_unit_norm_solver(field_data=analysis["field_data"])
        ures = solver.solve(ratioF)

        cand_diag["unit_target_log"] = ures.get("target_log")
        cand_diag["unit_solution_vector"] = ures.get("solution_vector")

        if not ures["success"]:
            cand_diag["reason"] = "UNIT_CORRECTION_UNRESOLVED"
            cand_diag["unit_failure_reason"] = ures.get("reason")
            continue

        eps = ures["epsilon"]

        try:
            corrected = gen / eps
        except Exception as e:
            cand_diag["reason"] = "UNIT_CORRECTION_APPLY_FAILED"
            cand_diag["unit_failure_reason"] = str(e)
            continue

        try:
            check = (corrected * corrected.conjugate()) / emb(M)
            if check != 1:
                cand_diag["reason"] = "UNIT_CORRECTION_VERIFY_FAILED"
                cand_diag["unit_failure_reason"] = f"post-correction ratio = {check}"
                continue
        except Exception as e:
            cand_diag["reason"] = "UNIT_CORRECTION_VERIFY_FAILED"
            cand_diag["unit_failure_reason"] = str(e)
            continue

        r = _root_from_field_generator(corrected, K)
        if r is None:
            cand_diag["reason"] = "CORRECTED_GENERATOR_NOT_INTEGRAL_BASIS"
            continue

        if r not in seen:
            seen.add(r)
            roots.append(r)

        cand_diag["reason"] = "UNIT_CORRECTED"
        
        
    if not roots and diagnostics["principal_candidate_count"] == 0:
        diagnostics["failure_reason"] = (
            "GENERATOR_EXTRACTION_FAILED" if saw_principal_report else "NO_PRINCIPAL_CANDIDATES"
        )
    elif not roots and diagnostics["generator_count"] == 0:
        diagnostics["failure_reason"] = "GENERATOR_EXTRACTION_FAILED"
    elif not roots:
        diagnostics["failure_reason"] = "UNIT_CORRECTION_FAILED"

    return roots, diagnostics


# ----------------------------------------------------------------------
# Top-level root search API
# ----------------------------------------------------------------------

def actual_roots_from_ideal_search(
    coeffs,
    *,
    orbit=True,
    check_real_embeddings=True,
    verbose=False,
    rank=None,
    coord=None,
    use_legacy_fallback=True,
    max_candidates=256,
):
    m0, m1, m2 = [int(c) for c in coeffs]
    screen = quick_screen_M(m0, m1, m2, check_real_embeddings=check_real_embeddings)
    B2 = screen.get("B2", B2_from_m012(m0, m1, m2))
    B = screen.get("B", int(math.isqrt(B2)) if B2 >= 0 else None)
    base = {
        "screen": screen["status"],
        "B2": B2,
        "B": B,
        "used_legacy": False,
        "status": None,
        "actual_roots": [],
        "diagnostics": {},
    }
    if screen["status"] == "ZERO":
        roots=[(0,0,0,0,0,0)]
        if orbit:
            roots=list(orbit_expand_roots(roots))
        base.update({"status": "EXACT_ROOT", "actual_roots": roots, "diagnostics": {"reason": "ZERO"}})
        return base
    if screen["status"] != "PASSES_IDEAL_SIEVE":
        base.update({"status": "SCREEN_REJECT", "diagnostics": {"reason": screen["status"]}})
        return base
    if m1 == 0 and m2 == 0 and m0 >= 0:
        s = math.isqrt(m0)
        if s * s == m0:
            roots=[(s,0,0,0,0,0)]
            if orbit:
                roots=list(orbit_expand_roots(roots))
            base.update({"status": "EXACT_ROOT", "actual_roots": roots, "diagnostics": {"reason": "RATIONAL_SQUARE"}})
            return base
    roots, diags = simple_constructive_roots((m0,m1,m2), max_candidates=max_candidates, verbose=verbose)
    if roots:
        if orbit:
            roots=list(orbit_expand_roots(roots))
        base.update({"status": "IDEAL_CONSTRUCTIVE", "actual_roots": roots, "diagnostics": diags})
        return base
    base["diagnostics"] = diags
    if not use_legacy_fallback:
        base["status"] = "NO_ROOTS_FOUND"
        return base
    if verbose:
        prefix = f"rank {rank} " if rank is not None else ""
        coord_str = f"coord={coord}, " if coord is not None else ""
        print(f"{prefix}entering legacy solver: {coord_str}Y={(m0,m1,m2)}")
    legacy_roots = direct_roots_via_legacy_backend((m0,m1,m2), verbose=verbose, rank=rank, orbit=orbit)
    base.update({"status": "LEGACY_RESULT", "used_legacy": True, "actual_roots": legacy_roots})
    return base


__all__ = [
    "build_fields", "real_embeddings_of_M", "classify_prime_in_extension", "classify_prime_in_extension_cached", "analyze_M",
    "quick_screen_M", "enumerate_candidate_ideals", "process_M", "process_many",
    "actual_roots_from_ideal_search", "factor_ideal_compat", "principal_ideal_factorization",
    "lift_K_to_F_if_possible", "unit_norm_preimage_in_K", "unit_is_global_norm_in_K_over_F", 
    "UnitNormSolver", "get_unit_norm_solver",
    "new_quick_screen_profile", "analyze_M_profiled", "quick_screen_status_profiled",
    "quick_screen_M_profiled", "analyze_M_fastscreen", "quick_screen_M_fast",
    "analyze_M_fastscreen_profiled", "quick_screen_status_fast_profiled",
    "quick_screen_M_fast_profiled", "_profile_summary",
]


# ----------------------------------------------------------------------
# Reporting helpers kept compatible
# ----------------------------------------------------------------------

def _format_prime_data(item):
    return {
        "P_norm": int(item["P_norm"]),
        "e": int(item["e"]),
        "kind": item["kind"],
        "P_repr": str(item["P"]),
        "ext_fac_repr": str(item["ext_fac"]),
    }


def process_M(coeffs, prec=100, max_candidates=None, field_data=None):
    analysis = analyze_M(coeffs, prec=prec, field_data=field_data)
    candidates = []
    if analysis["status"] == "PASSES_IDEAL_SIEVE":
        candidates = enumerate_candidate_ideals(analysis, max_candidates=max_candidates)
    return {
        "input": analysis["input"],
        "status": analysis["status"],
        "all_real_nonnegative": analysis["all_real_nonnegative"],
        "real_embeddings": [str(v) for v in analysis["real_embeddings"]],
        "prime_data": [_format_prime_data(item) for item in analysis["prime_data"]],
        "candidate_count": len(candidates),
        "candidates": [
            # cand["ideal_hnf"] is a PARI HNF (audit B); stringify for diag.
            {"ideal_hnf": str(item.get("ideal_hnf", item.get("ideal"))),
             "local_choice_summary": [str(x) for x in item["local_choice_summary"]]}
            for item in candidates
        ],
    }


def process_many(values, prec=100, max_candidates=None):
    field_data = build_fields()
    return [process_M(v, prec=prec, max_candidates=max_candidates, field_data=field_data) for v in values]


# ----------------------------------------------------------------------
# Internal profiling helpers for quick_screen_M / analyze_M
# ----------------------------------------------------------------------

def new_quick_screen_profile():
    return {
        "calls": 0,
        "t_build_M": 0.0,
        "t_real_embeddings": 0.0,
        "t_factorization_F": 0.0,
        "t_classify_extension": 0.0,
        "t_local_p3": 0.0,
        "t_finalize": 0.0,
        "t_total": 0.0,
        "classify_cache_profile": new_classify_cache_profile(),
        "status_counts": {},
    }


def _profile_inc_status(profile, status):
    d = profile["status_counts"]
    d[status] = d.get(status, 0) + 1


def analyze_M_profiled(coeffs, prec=80, field_data=None, profile=None):
    t0 = time.perf_counter() if profile is not None else None

    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data

    m0, m1, m2 = [ZZ(c) for c in coeffs]

    tb0 = time.perf_counter() if profile is not None else None
    M = F(m0 + m1 * alpha + m2 * alpha**2)
    if profile is not None:
        profile["t_build_M"] += time.perf_counter() - tb0

    if m0 == 0 and m1 == 0 and m2 == 0:
        out = {
            "input": (m0, m1, m2),
            "M": M,
            "real_embeddings": [ZZ(0), ZZ(0), ZZ(0)],
            "all_real_nonnegative": True,
            "factorization_F": [],
            "prime_data": [],
            "inert_odd_obstruction": False,
            "status": "ZERO",
            "field_data": field_data,
            "B2": 0,
            "B": 0,
            "split_prime_count": 0,
            "inert_prime_count": 0,
            "ramified_prime_count": 0,
        }
        if profile is not None:
            profile["calls"] += 1
            profile["t_total"] += time.perf_counter() - t0
            _profile_inc_status(profile, "ZERO")
        return out

    te0 = time.perf_counter() if profile is not None else None
    real_vals = real_embeddings_of_M(F, M, prec=prec)
    if profile is not None:
        profile["t_real_embeddings"] += time.perf_counter() - te0

    tf0 = time.perf_counter() if profile is not None else None
    fac = principal_ideal_factorization(F, M)
    if profile is not None:
        profile["t_factorization_F"] += time.perf_counter() - tf0

    prime_data = []
    inert_odd = False
    tc0 = time.perf_counter() if profile is not None else None
    for P, e in fac:
        e_int = int(e)

        kind, ext_fac = classify_prime_in_extension_cached(
            P, K, emb,
            cache_profile=profile.get("classify_cache_profile") if profile is not None else None,
        )
        
        item = {
            "P": P,
            "e": e_int,
            "P_norm": ZZ(P.norm()),
            "kind": kind,
            "ext_fac": ext_fac,
        }
        prime_data.append(item)
        if kind == "inert" and (e_int % 2 == 1):
            inert_odd = True
    if profile is not None:
        profile["t_classify_extension"] += time.perf_counter() - tc0

    tfz0 = time.perf_counter() if profile is not None else None
    split_count = sum(1 for x in prime_data if x["kind"] == "split")
    inert_count = sum(1 for x in prime_data if x["kind"] == "inert")
    ram_count = sum(1 for x in prime_data if x["kind"] == "ramified")
    status = "REJECT_INERT_ODD" if inert_odd else "PASSES_IDEAL_SIEVE"
    B2 = B2_from_m012(m0, m1, m2)
    out = {
        "input": (m0, m1, m2),
        "M": M,
        "real_embeddings": real_vals,
        "all_real_nonnegative": all(v >= 0 for v in real_vals),
        "factorization_F": fac,
        "prime_data": prime_data,
        "inert_odd_obstruction": inert_odd,
        "status": status,
        "field_data": field_data,
        "B2": B2,
        "B": int(math.isqrt(B2)) if B2 >= 0 else None,
        "split_prime_count": split_count,
        "inert_prime_count": inert_count,
        "ramified_prime_count": ram_count,
    }
    if profile is not None:
        profile["t_finalize"] += time.perf_counter() - tfz0
        profile["calls"] += 1
        profile["t_total"] += time.perf_counter() - t0
        _profile_inc_status(profile, status)
    return out


def quick_screen_status_profiled(
    m0, m1, m2,
    check_real_embeddings=False,
    prec=80,
    field_data=None,
    check_local_p3k=False,
    local_p3k=2,
    profile=None,
):
    t0 = time.perf_counter() if profile is not None else None
    analysis = analyze_M_profiled((m0, m1, m2), prec=prec, field_data=field_data, profile=profile)

    if analysis["status"] == "ZERO":
        status = "ZERO"
    elif check_real_embeddings and not analysis["all_real_nonnegative"]:
        status = "REJECT_NEGATIVE_EMBEDDING"
    else:
        status = analysis["status"]

    if status == "PASSES_IDEAL_SIEVE" and check_local_p3k:
        tp0 = time.perf_counter() if profile is not None else None
        p3scr = quick_local_p3k_norm_screen(
            m0, m1, m2,
            field_data=analysis["field_data"],
            k=local_p3k,
        )
        if profile is not None:
            profile["t_local_p3"] += time.perf_counter() - tp0
        if not p3scr["passes"]:
            status = "REJECT_LOCAL_P3K"

    if profile is not None:
        # analyze_M_profiled already incremented status; correct if overridden here
        last = analysis["status"]
        if status != last:
            d = profile["status_counts"]
            d[last] = d.get(last, 1) - 1
            if d[last] <= 0:
                d.pop(last, None)
            d[status] = d.get(status, 0) + 1
            profile["t_total"] += time.perf_counter() - t0 - 0.0
    return status


def quick_screen_M_profiled(
    m0, m1, m2,
    check_real_embeddings=False,
    prec=80,
    field_data=None,
    check_local_p3k=False,
    local_p3k=2,
    profile=None,
):
    analysis = analyze_M_profiled((m0, m1, m2), prec=prec, field_data=field_data, profile=profile)

    if analysis["status"] == "ZERO":
        status = "ZERO"
    elif check_real_embeddings and not analysis["all_real_nonnegative"]:
        status = "REJECT_NEGATIVE_EMBEDDING"
    else:
        status = analysis["status"]

    if status == "PASSES_IDEAL_SIEVE" and check_local_p3k:
        tp0 = time.perf_counter() if profile is not None else None
        p3scr = quick_local_p3k_norm_screen(
            m0, m1, m2,
            field_data=analysis["field_data"],
            k=local_p3k,
        )
        if profile is not None:
            profile["t_local_p3"] += time.perf_counter() - tp0
        if not p3scr["passes"]:
            status = "REJECT_LOCAL_P3K"

    return {
        "input": analysis["input"],
        "status": status,
        "split_prime_count": analysis["split_prime_count"],
        "inert_prime_count": analysis["inert_prime_count"],
        "ramified_prime_count": analysis["ramified_prime_count"],
        "all_real_nonnegative": analysis["all_real_nonnegative"],
        "B2": analysis["B2"],
        "B": analysis["B"],
    }




def analyze_M_fastscreen_profiled(coeffs, prec=80, field_data=None, profile=None):
    t0 = time.perf_counter() if profile is not None else None

    if field_data is None:
        field_data = build_fields()
    F, OF, K, OK, alpha, z, emb = field_data

    m0, m1, m2 = [ZZ(c) for c in coeffs]

    tb0 = time.perf_counter() if profile is not None else None
    M = F(m0 + m1 * alpha + m2 * alpha**2)
    if profile is not None:
        profile["t_build_M"] += time.perf_counter() - tb0

    if m0 == 0 and m1 == 0 and m2 == 0:
        out = {
            "input": (m0, m1, m2),
            "M": M,
            "real_embeddings": [ZZ(0), ZZ(0), ZZ(0)],
            "all_real_nonnegative": True,
            "factorization_F": [],
            "inert_odd_obstruction": False,
            "status": "ZERO",
            "field_data": field_data,
            "B2": 0,
            "B": 0,
            "split_prime_count": 0,
            "inert_prime_count": 0,
            "ramified_prime_count": 0,
        }
        if profile is not None:
            profile["calls"] += 1
            profile["t_total"] += time.perf_counter() - t0
            _profile_inc_status(profile, "ZERO")
        return out

    te0 = time.perf_counter() if profile is not None else None
    real_vals = real_embeddings_of_M(F, M, prec=prec)
    if profile is not None:
        profile["t_real_embeddings"] += time.perf_counter() - te0

    tf0 = time.perf_counter() if profile is not None else None
    fac = principal_ideal_factorization(F, M)
    if profile is not None:
        profile["t_factorization_F"] += time.perf_counter() - tf0

    split_count = 0
    inert_count = 0
    ram_count = 0
    inert_odd = False
    tc0 = time.perf_counter() if profile is not None else None
    for P, e in fac:
        e_int = int(e)
        kind = classify_prime_in_extension_from_norm(P)
        if kind == "split":
            split_count += 1
        elif kind == "inert":
            inert_count += 1
            if (e_int % 2) == 1:
                inert_odd = True
        elif kind == "ramified":
            ram_count += 1
        else:
            raise RuntimeError(f"unexpected fast screen classification: {kind}")
    if profile is not None:
        profile["t_classify_extension"] += time.perf_counter() - tc0

    tfz0 = time.perf_counter() if profile is not None else None
    status = "REJECT_INERT_ODD" if inert_odd else "PASSES_IDEAL_SIEVE"
    B2 = B2_from_m012(m0, m1, m2)
    out = {
        "input": (m0, m1, m2),
        "M": M,
        "real_embeddings": real_vals,
        "all_real_nonnegative": all(v >= 0 for v in real_vals),
        "factorization_F": fac,
        "inert_odd_obstruction": inert_odd,
        "status": status,
        "field_data": field_data,
        "B2": B2,
        "B": int(math.isqrt(B2)) if B2 >= 0 else None,
        "split_prime_count": split_count,
        "inert_prime_count": inert_count,
        "ramified_prime_count": ram_count,
    }
    if profile is not None:
        profile["t_finalize"] += time.perf_counter() - tfz0
        profile["calls"] += 1
        profile["t_total"] += time.perf_counter() - t0
        _profile_inc_status(profile, status)
    return out


def quick_screen_status_fast_profiled(
    m0, m1, m2,
    check_real_embeddings=False,
    prec=80,
    field_data=None,
    check_local_p3k=False,
    local_p3k=2,
    profile=None,
):
    t0 = time.perf_counter() if profile is not None else None
    analysis = analyze_M_fastscreen_profiled((m0, m1, m2), prec=prec, field_data=field_data, profile=profile)

    if analysis["status"] == "ZERO":
        status = "ZERO"
    elif check_real_embeddings and not analysis["all_real_nonnegative"]:
        status = "REJECT_NEGATIVE_EMBEDDING"
    else:
        status = analysis["status"]

    if status == "PASSES_IDEAL_SIEVE" and check_local_p3k:
        tp0 = time.perf_counter() if profile is not None else None
        p3scr = quick_local_p3k_norm_screen(
            m0, m1, m2,
            field_data=analysis["field_data"],
            k=local_p3k,
        )
        if profile is not None:
            profile["t_local_p3"] += time.perf_counter() - tp0
        if not p3scr["passes"]:
            status = "REJECT_LOCAL_P3K"

    if profile is not None:
        last = analysis["status"]
        if status != last:
            d = profile["status_counts"]
            d[last] = d.get(last, 1) - 1
            if d[last] <= 0:
                d.pop(last, None)
            d[status] = d.get(status, 0) + 1
            profile["t_total"] += time.perf_counter() - t0 - 0.0
    return status


def quick_screen_M_fast_profiled(
    m0, m1, m2,
    check_real_embeddings=False,
    prec=80,
    field_data=None,
    check_local_p3k=False,
    local_p3k=2,
    profile=None,
):
    analysis = analyze_M_fastscreen_profiled((m0, m1, m2), prec=prec, field_data=field_data, profile=profile)

    if analysis["status"] == "ZERO":
        status = "ZERO"
    elif check_real_embeddings and not analysis["all_real_nonnegative"]:
        status = "REJECT_NEGATIVE_EMBEDDING"
    else:
        status = analysis["status"]

    if status == "PASSES_IDEAL_SIEVE" and check_local_p3k:
        tp0 = time.perf_counter() if profile is not None else None
        p3scr = quick_local_p3k_norm_screen(
            m0, m1, m2,
            field_data=analysis["field_data"],
            k=local_p3k,
        )
        if profile is not None:
            profile["t_local_p3"] += time.perf_counter() - tp0
        if not p3scr["passes"]:
            status = "REJECT_LOCAL_P3K"

    return {
        "input": analysis["input"],
        "status": status,
        "split_prime_count": analysis["split_prime_count"],
        "inert_prime_count": analysis["inert_prime_count"],
        "ramified_prime_count": analysis["ramified_prime_count"],
        "all_real_nonnegative": analysis["all_real_nonnegative"],
        "B2": analysis["B2"],
        "B": analysis["B"],
    }

def new_classify_cache_profile():
    return {
        "calls": 0,
        "hits": 0,
        "misses": 0,
        "t_key": 0.0,
        "t_hit_lookup": 0.0,
        "t_miss_work": 0.0,
    }
    

def _profile_summary(profile):
    lines = []
    lines.append("quick_screen_M profiling summary:")
    lines.append(f"  calls               : {profile.get('calls', 0)}")
    lines.append(f"  t_build_M           : {profile.get('t_build_M', 0.0):.6f} s")
    lines.append(f"  t_real_embeddings   : {profile.get('t_real_embeddings', 0.0):.6f} s")
    lines.append(f"  t_factorization_F   : {profile.get('t_factorization_F', 0.0):.6f} s")
    lines.append(f"  t_classify_extension: {profile.get('t_classify_extension', 0.0):.6f} s")
    lines.append(f"  t_local_p3          : {profile.get('t_local_p3', 0.0):.6f} s")
    lines.append(f"  t_finalize          : {profile.get('t_finalize', 0.0):.6f} s")
    lines.append(f"  t_total             : {profile.get('t_total', 0.0):.6f} s")
    lines.append("  status_counts:")
    for k in sorted(profile.get("status_counts", {})):
        lines.append(f"    {k}: {profile['status_counts'][k]}")

    cc = profile.get("classify_cache_profile", {})
    if cc:
        lines.append("  classify_cache:")
        lines.append(f"    calls      : {cc.get('calls', 0)}")
        lines.append(f"    hits       : {cc.get('hits', 0)}")
        lines.append(f"    misses     : {cc.get('misses', 0)}")
        lines.append(f"    hit rate   : {100.0 * cc.get('hits', 0) / max(1, cc.get('calls', 0)):.2f}%")
        lines.append(f"    t_key      : {cc.get('t_key', 0.0):.6f} s")
        lines.append(f"    t_hit      : {cc.get('t_hit_lookup', 0.0):.6f} s")
        lines.append(f"    t_miss     : {cc.get('t_miss_work', 0.0):.6f} s")
        
    return "\n".join(lines)
