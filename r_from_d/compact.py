"""compact.py <in.json> <out.json>: merge the Clifford segments between T gates and re-synthesize each
with a BFS-shortest symplectic word (+ Pauli); global phases dropped (verified up to global phase)."""
import sys, json, numpy as np
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/r_from_d')
from build_circuit import word_mat, clifford_word, sympl_table
d = json.load(open(sys.argv[1])); gates = [tuple(g) for g in d['gates']]
tab = sympl_table(); out = []; seg = []
for g in gates + [('END', None)]:
    if g[0] in ('T', 'Tdg', 'END'):
        if seg:
            wd, lam = clifford_word(word_mat(seg), tab); out += wd
        seg = []
        if g[0] != 'END': out.append(g)
    else: seg.append(g)
d['gates'] = [list(g) for g in out]; json.dump(d, open(sys.argv[2], 'w'))
print(len(gates), '->', len(out), 'gates;', ' '.join(g[0]+('' if g[1] is None else str(g[1])) for g in out))
