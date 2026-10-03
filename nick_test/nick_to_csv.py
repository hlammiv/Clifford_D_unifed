"""nick_to_csv.py — convert ingest_decompose.py's JSON output into the
HRSA-schema CSV that plot_nd_vs_eps_v2.py consumes.

Schema: theta, epsilon, method, N_D, achieved_frob, all_checks_pass
  - method     = "nick(f=N)"
  - epsilon    = achieved_frob (these are exact-ring fits; the achieved
                 precision IS the spec, so target == achieved)
  - all_checks_pass = exact-unitary AND decompose-success AND reconstruction-OK

Usage: python3 nick_to_csv.py nick_decompositions.json out.csv
"""
import csv
import json
import sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)

rows = []
for fname, blob in data.items():
    f = blob["f"]
    for rec in blob["matrices"]:
        d = rec.get("decompose", {})
        ok = (rec["unitary_exact"] and d.get("success")
              and d.get("reconstruction_matches"))
        if not (ok and isinstance(d.get("N_D"), int)):
            continue
        rows.append({
            "theta": rec["theta"],
            "epsilon": rec["achieved_frob"],
            "method": f"nick(f={f})",
            "N_D": d["N_D"],
            "achieved_frob": rec["achieved_frob"],
            "all_checks_pass": "true",
        })

with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["theta", "epsilon", "method", "N_D",
                                       "achieved_frob", "all_checks_pass"])
    w.writeheader()
    w.writerows(rows)

print(f"wrote {sys.argv[2]}: {len(rows)} cells")
