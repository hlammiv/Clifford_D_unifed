"""--topk 0 (default) regression: patched fit_one_theta_all_best == main's, same stubs, several thetas/atol."""
import sys, os, json, tempfile, importlib
import numpy as np
exec(open("test_patch.py").read().split("# ---------------- A ----------------")[0])   # imports + pool
sys.path.insert(0, "origpkg/zeta9")
import importlib.util
_sp = importlib.util.spec_from_file_location("orig_fit", "origpkg/zeta9/orig_fit.py")
O = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(O)
exec(open("test_patch.py").read().split("# ---------------- synthetic root lists ----------------")[1].split("kw = dict(")[0])
tmp = tempfile.mkdtemp(dir=".")
tri_path = os.path.join(tmp, "triples"); tri_rows.tofile(tri_path)
json.dump({"rows_written": int(len(tri_rows))}, open(tri_path + ".manifest.json", "w"))
prefix = os.path.join(tmp, "roots_local")
for mod in (M, O):
    open(mod._state_paths(prefix, 512)["exact_roots_index_meta"], "w").write("{}")
    mod._read_triple_rows = lambda path, start, n: tri_rows[start:start + n]
    mod._build_locator = lambda needed, paths: {}
    mod._load_fixed_target_cache = lambda *a, **k: (np.ascontiguousarray(tri_rows[:, 3:6]), {})
    mod._ensure_binned_phase_sidecar = lambda *a, **k: {}
    mod._isin_structured_rows = lambda rows, keys: np.ones(rows.shape[0], dtype=bool)
n = 0
for th in (0.3, 0.7, 1.9, 2.8):
    for atol in (1e-24, 1e-12):
        t1 = complex(amp * np.exp(-0.5j * th))
        for mod in (M, O):
            mod._candidate_cache_get = lambda cache, y, target, *a, **k: cand_list(y, target)
        outs = []
        for mod, tag in ((M, "new"), (O, "old")):
            mod.fit_one_theta_all_best(triples_file=tri_path, triples_json=tri_path + ".manifest.json", rootdb_prefix=prefix,
                f=F, theta=th, eps=10.0, output_prefix=os.path.join(tmp, tag), triples_chunk_rows=5,
                y2_good_npz=tri_path, norm="2/3^f", equal_atol=atol)
            outs.append(np.load(os.path.join(tmp, tag + ".theta_all_best.npz")))
        for k in outs[0].files:
            assert np.array_equal(outs[0][k], outs[1][k], equal_nan=True), (th, atol, k)
        n += 1
        print(f"theta={th} atol={atol:g}: identical .theta_all_best.npz (n_solutions={len(outs[0]['coeffs'])})")
print("REGRESSION OK", n)
