"""Run original or fast decomposition on a fixed matrix set; dump JSON lines.
  python3 verify.py --impl orig --out orig.jsonl
  python3 verify.py --impl fast --out fast.jsonl
  python3 verify.py --compare orig.jsonl fast.jsonl
"""
import argparse, json, sys, time
from pathlib import Path
D = Path(__file__).resolve().parent
U = D.parent
sys.path[:0] = [str(D), str(U / "nick_test"), str(U / "nick_request"), str(U), str(U / "hrsa")]


def matrix_set(per_fits=4, per_topk=2, tcost_extra=1):
    from ingest_decompose import parse_fits_file
    from analyze_topk import parse_topk_file
    items = []
    for f in (4, 6, 8, 10, 12, 14, 16):
        F, rows = parse_fits_file(U / "nick_test" / f"fits_f={f}.txt")
        step = max(1, len(rows) // per_fits)
        for r in rows[::step][:per_fits]:
            items.append(dict(src=f"fits_f={f}:{r[0]}", f=F, theta=r[1], g=r[2], select="d"))
        F, rows = parse_topk_file(U / "nick_request/topk_run/samples" / f"fits_topk_f={f}.sample30.txt")
        step = max(1, len(rows) // per_topk)
        for r in rows[step // 2::step][:per_topk]:
            g = [r["groups"][3 * (k % 3) + k // 3] for k in range(9)]   # --colmajor as in production
            items.append(dict(src=f"topk_f={f}:L{r['line']}:colmajor", f=F, theta=r["theta"], g=g, select="d"))
        for r in rows[step // 3::step][:tcost_extra]:
            items.append(dict(src=f"topk_f={f}:L{r['line']}:tcost", f=F, theta=r["theta"], g=r["groups"], select="tcost"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["orig", "fast"])
    ap.add_argument("--out")
    ap.add_argument("--fs", default="4,6,8,10,12,14,16")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        A = {r["src"]: r for r in map(json.loads, open(a.compare[0]))}
        B = {r["src"]: r for r in map(json.loads, open(a.compare[1]))}
        common = [k for k in A if k in B]
        bad = [k for k in common if A[k]["result"] != B[k]["result"]]
        print(f"{len(common)} common matrices, {len(bad)} mismatches")
        for k in bad:
            print("MISMATCH", k)
        from collections import defaultdict
        tA, tB = defaultdict(list), defaultdict(list)
        for k in common:
            tA[A[k]["f"]].append(A[k]["wall"]); tB[B[k]["f"]].append(B[k]["wall"])
        print(f"{'f':>3} {'n':>3} {'orig s/mat':>11} {'fast s/mat':>11} {'speedup':>8}")
        for f in sorted(tA):
            mo, mf = sum(tA[f]) / len(tA[f]), sum(tB[f]) / len(tB[f])
            print(f"{f:>3} {len(tA[f]):>3} {mo:>11.2f} {mf:>11.3f} {mo / mf:>8.1f}")
        return
    import canonical_reducer as cr
    import nick_tcost_all
    captured = {}
    if a.impl == "fast":
        import fast_decompose as fd
        base = fd.decompose_fast
    else:
        base = cr.decompose_canonical
        t = time.time(); cr.get_prefix_table(); print(f"table build {time.time()-t:.1f}s", flush=True)
    def wrapped(V, **kw):
        r = base(V, **kw); captured["r"] = r; return r
    cr.decompose_canonical = wrapped
    fs = set(map(int, a.fs.split(",")))
    with open(a.out, "w") as fh:
        for it in matrix_set():
            if it["f"] not in fs:
                continue
            t = time.time()
            rec = nick_tcost_all.work((it["f"], it["theta"], it["g"], it["select"]))
            wall = time.time() - t
            r = captured["r"]
            tc = [[[int(c) for c in e.num.coefs] + [e.denom_pow3] for e in row] for row in r["trailing_clifford"]]
            res = dict(rec=rec, syllables=r["syllables"], D_count=r["D_count"], n_iter=r["n_iter"],
                       success=r["success"], trailing=tc, error=r.get("error"))
            fh.write(json.dumps(dict(src=it["src"], f=it["f"], wall=wall, result=res)) + "\n")
            fh.flush()
            print(f"{it['src']:<40} wall={wall:8.2f}s N_D={rec.get('N_D')} Tcost={rec.get('Tcost')}", flush=True)
    cr.set_selection_cost("d")


if __name__ == "__main__":
    main()
