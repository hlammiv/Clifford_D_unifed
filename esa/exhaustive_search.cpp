#include "exhaustive_search.h"
#include <chrono>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <unordered_map>

using namespace std;

const double FPRL = 0.0001;
extern std::atomic<bool> interrupted;

// Fix J (2026-05): trig constants for inline Re/Im computation matching
// ringZ9::real_part() / imag_part().  Same fold of element[4]±element[5]
// using the cofunction symmetries cos(8π/9) = cos(10π/9) = -cos(π/9) and
// sin(8π/9) = -sin(10π/9) = sin(π/9 reflected).  Keep these as the SINGLE
// source of truth; if cyclotomic_int9.cpp's constants change, update here too.
namespace {
	const double JFIX_RCOS[4] = {
		 0.766044443118978,    // cos(2π/9)
		 0.17364817766693041,  // cos(4π/9) = sin(π/18)
		-0.5,                  // cos(6π/9)
		-0.9396926207859083    // cos(8π/9) = cos(10π/9)
	};
	const double JFIX_RSIN[4] = {
		 0.6427876096865393,   // sin(2π/9)
		 0.984807753012208,    // sin(4π/9)
		 0.8660254037844387,   // sin(6π/9)
		 0.3420201433256689    // sin(8π/9) = -sin(10π/9)
	};
}

// =====================================================================
// ESA_DIAG: per-f cascade counters for the inner-loop hot path.
// Compiled in only with -DESA_DIAG; otherwise all DIAG_* macros are no-ops.
// =====================================================================
#ifdef ESA_DIAG
#include <signal.h>
namespace {
	struct ESADiag {
		std::atomic<long long> x3_entries{0};
		std::atomic<long long> x3_passed{0};      // x_3 reaching the y_2 loop
		std::atomic<long long> y2_iters{0};
		std::atomic<long long> y2_passed{0};      // y_2 reaching the udc call
		std::atomic<long long> udc_calls{0};
		std::atomic<long long> udc_pass{0};
		std::atomic<long long> eCU_calls{0};
		std::atomic<long long> eCU_pairs{0};      // (i,j) iterations inside eCU
		std::atomic<long long> y1_calls{0}, y1_pass{0};
		std::atomic<long long> z2_calls{0}, z2_pass{0};
		std::atomic<long long> z1_calls{0}, z1_pass{0};
		std::atomic<long long> isUnit_calls{0}, isUnit_pass{0};
		std::atomic<long long> chkEps_calls{0}, chkEps_pass{0};

		void reset(){
			x3_entries = 0; x3_passed = 0;
			y2_iters = 0; y2_passed = 0;
			udc_calls = 0; udc_pass = 0;
			eCU_calls = 0; eCU_pairs = 0;
			y1_calls = 0; y1_pass = 0;
			z2_calls = 0; z2_pass = 0;
			z1_calls = 0; z1_pass = 0;
			isUnit_calls = 0; isUnit_pass = 0;
			chkEps_calls = 0; chkEps_pass = 0;
		}

		void print(int f){
			auto pct = [](long long n, long long d){
				return d > 0 ? 100.0 * (double)n / (double)d : 0.0;
			};
			fprintf(stderr,
				"[f=%d ESA-DIAG]\n"
				"  x_3 loop:   entries=%lld  pass_filters=%lld (%.4f%%)\n"
				"  y_2 loop:   iters=%lld    pass_eps=%lld    (%.4f%%)\n"
				"  unitDiag:   calls=%lld    pass=%lld        (%.4f%%)\n"
				"  eCU:        calls=%lld    pair_iters=%lld\n"
				"  solveSys y1: %lld pass / %lld calls (%.6f%%)\n"
				"  solveSys z2: %lld pass / %lld calls (%.6f%%)\n"
				"  solveSys z1: %lld pass / %lld calls (%.6f%%)\n"
				"  isUnitary:   %lld pass / %lld calls (%.4f%%)\n"
				"  checkEps:    %lld pass / %lld calls (%.4f%%)\n",
				f,
				(long long)x3_entries, (long long)x3_passed,
				pct(x3_passed, x3_entries),
				(long long)y2_iters, (long long)y2_passed,
				pct(y2_passed, y2_iters),
				(long long)udc_calls, (long long)udc_pass,
				pct(udc_pass, udc_calls),
				(long long)eCU_calls, (long long)eCU_pairs,
				(long long)y1_pass, (long long)y1_calls,
				pct(y1_pass, y1_calls),
				(long long)z2_pass, (long long)z2_calls,
				pct(z2_pass, z2_calls),
				(long long)z1_pass, (long long)z1_calls,
				pct(z1_pass, z1_calls),
				(long long)isUnit_pass, (long long)isUnit_calls,
				pct(isUnit_pass, isUnit_calls),
				(long long)chkEps_pass, (long long)chkEps_calls,
				pct(chkEps_pass, chkEps_calls));
		}
	};
	ESADiag g_diag;
	std::atomic<int> g_diag_f{-1};   // current f, set before each parallel-for

	void diag_signal_handler(int){
		int f = g_diag_f.load(std::memory_order_relaxed);
		if(f >= 0) g_diag.print(f);
	}

	struct DiagSigInstaller {
		DiagSigInstaller(){
			struct sigaction sa{};
			sa.sa_handler = diag_signal_handler;
			sigemptyset(&sa.sa_mask);
			sa.sa_flags = SA_RESTART;
			sigaction(SIGUSR1, &sa, nullptr);
			fprintf(stderr,
				"[ESA_DIAG] PID=%d  send SIGUSR1 (kill -USR1 %d) for live diag dump\n",
				(int)getpid(), (int)getpid());
		}
	};
	DiagSigInstaller g_diag_sig_installer;
}
#define DIAG_INC(x)   (g_diag.x.fetch_add(1, std::memory_order_relaxed))
#define DIAG_RESET()  g_diag.reset()
#define DIAG_PRINT(f) g_diag.print(f)
#define DIAG_SET_F(f) g_diag_f.store((f), std::memory_order_relaxed)
#define DIAG_OMP_SHARED , g_diag
#else
#define DIAG_INC(x)   ((void)0)
#define DIAG_RESET()  ((void)0)
#define DIAG_PRINT(f) ((void)0)
#define DIAG_SET_F(f) ((void)0)
#define DIAG_OMP_SHARED
#endif

// -----------------------------------------------------------------------
// Hash for ringZ9 by its 6-coefficient canonical representation.
// Used to memoize fieldNorm and partialFieldNorm inside solveSystem.
// -----------------------------------------------------------------------
struct RingZ9Hash {
	size_t operator()(const array<int,6>& a) const noexcept {
		size_t h = 0;
		for(int v : a) h ^= hash<int>{}(v) + 0x9e3779b9u + (h<<6) + (h>>2);
		return h;
	}
};

// Thread-local caches cleared once per ESA call via clearSolveSystemCache().
// solveSystem is called with a small set of distinct denominators per search,
// so the cache stays tiny and lookup is O(1).
static thread_local unordered_map<array<int,6>, int,        RingZ9Hash> s_fieldNormCache;
static thread_local unordered_map<array<int,6>, ringZ9,     RingZ9Hash> s_partialNormCache;

static void clearSolveSystemCache(){
	s_fieldNormCache.clear();
	s_partialNormCache.clear();
}

// -----------------------------------------------------------------------
// ESA  (optimised)
// Changes vs original:
//   - Loop-invariant subexpressions hoisted out of inner loops
//   - y_2 loop moved OUTSIDE x_3 loop (y_2 is independent of x_3)
//   - pro_lookup vector pre-allocated and reused (no per-pair alloc)
//   - abs_val_sq comparisons replace ring-multiply == ring-element checks
//     in copy_if lambdas
//   - x1x1dag / z3z3dag stored as double (they are real non-neg integers)
//   - All debug cout removed (guard with #ifdef DEBUG if needed)
// -----------------------------------------------------------------------
array<ringZ9chi, 9> ESA(double theta, double epsilon, int max_f){
	// Convention reconciliation: this file's internal logic was written for the
	// convention V[0][0] ~ e^{+i theta/2}, V[1][1] ~ e^{-i theta/2}, i.e. the
	// matrix R^Z_{(0,1)}(-theta). The unified compiler convention (per the paper
	// in ESA_CliffordD_Notes) is target = Diag(e^{-i theta/2}, e^{+i theta/2}, 1)
	// = R^Z_{(0,1)}(+theta). Negating theta on entry reconciles them without
	// touching every internal sin/cos.
	theta = -theta;

	int f = 0;
	array<vector<ringZ9>,3> cands;
	vector<ringZ9> lookup;

	while(f <= max_f){
		// --- per-f invariants ---
		const double f_pow    = pow(3, f);
		const double f_pow_sq = pow(3, 2*f);          // keep as double; int overflows at f>=10
		const int    f_pow_sq_i = three_power(2*f);
		ringZ9 rhs(f_pow_sq_i);
		clearSolveSystemCache();

#ifdef DEBUG
		cout << "f: " << f << endl;
#endif

		// Fix P (2026-05): max_candidates=0 → keep ALL diag candidates (no
		// top-500 truncation).  The truncation silently dropped HRSA's V's
		// diagonals at (theta=0.5, eps=0.05, f=4) — verified by esa_v_check.
		fullDiagEnumeration(cands, theta, epsilon, f, /*max_candidates=*/0);

#ifdef DEBUG
		cout << "Diagonals Enumerated. sizes: "
		     << cands[0].size() << " " << cands[1].size() << " " << cands[2].size() << endl;
#endif

		int minQ = min(findMinQ(cands[0], f), findMinQ(cands[2], f));

		fullX3Enumeration(lookup, f, minQ, epsilon);

#ifdef DEBUG
		cout << "Lookup size: " << lookup.size() << endl;
#endif

		// Precompute per-lookup-entry quantities.
		// abs_val_sq: floating-point |x|^2 for ε comparisons.
		// quad: integer-valued q-form (pro_map key, Fix F integer-arithmetic).
		// conj: complex-conjugate ring element (Fix G.1: hoisted out of fillUpper).
		vector<double> lookup_absq(lookup.size());
		vector<int>    lookup_quad(lookup.size());
		vector<ringZ9> lookup_conj(lookup.size());
		for(size_t i = 0; i < lookup.size(); i++){
			lookup_absq[i] = lookup[i].abs_val_sq();
			lookup_quad[i] = lookup[i].quad();
			lookup_conj[i] = lookup[i].complexConj();
		}

		// --------------------------------------------------------------------
		//  Fix B (2026-05): build pro_map ONCE per f, not once per (x_1,z_3).
		//
		//  pro_map keys an integer norm r and returns the vector of lookup
		//  entries with quad(x) == r.  Buckets hold values so we can pass
		//  &it->second straight to exhaustiveCompleteUnitary without copies.
		// --------------------------------------------------------------------
		unordered_map<int, vector<ringZ9>> pro_map;
		pro_map.reserve(lookup.size());
		for(size_t li = 0; li < lookup.size(); li++){
			pro_map[lookup_quad[li]].push_back(lookup[li]);
		}

		// Fix F (2026-05): rhs.getTerm(0) is the integer-valued constant
		// term of the rhs ring element.  rhs is constructed as ringZ9(f_pow_sq_i)
		// which sets element[0] = f_pow_sq_i and the rest 0, so rhs0 = 3^{2f}.
		const int rhs0 = rhs.getTerm(0);

		const double eps_sq   = epsilon * epsilon;

		for(const ringZ9& x_1 : cands[0]){
			if(interrupted) goto endthis;

			const double c_1      = abs(complex(cos(theta/2), sin(theta/2)) - x_1.toComplexDouble()/f_pow);
			const double c1_sq    = c_1 * c_1;
			if(c1_sq > eps_sq) continue;

			const double x1_abs_sq = x_1.abs_val_sq();
			// Fix F (2026-05): the only place x1x1dag was ever used was the
			// computation of r1 = (rhs - x1x1dag - x3x3dag).getTerm(0).  Since
			// (x_1 * x_1.complexConj()).getTerm(0) == x_1.quad() (verified by
			// ringZ9_unit_test invariant 14), and operator- is element-wise,
			// we can replace the ring construction with one integer:
			const int x1q = x_1.quad();
			// Fix G.1: precompute x_1 conjugate once per outer iteration; it
			// is reused for every (z_3, x_3, y_2) in the inner loops.
			const ringZ9 x1conj = x_1.complexConj();

			// --- y_2 pre-filter: store c2_sq alongside y_2 to avoid recomputation ---
			vector<pair<ringZ9,double>> y2_valid;
			y2_valid.reserve(cands[1].size());
			for(const ringZ9& y_2 : cands[1]){
				const double c_2   = abs(complex(cos(theta/2), -sin(theta/2)) - y_2.toComplexDouble()/f_pow);
				const double c2_sq = c_2 * c_2;
				if(c1_sq + c2_sq > eps_sq + FPRL) continue;
				// PRE-FIX BUG (2026-05): the original code called
				//   unitaryDiagCheck(x_1, y_2, ringZ9(0), f_pow)
				// here as a pre-filter, using z_3 = 0 as a "wildcard".  But the
				// triangle inequality |x_1| + |y_2| - |z_3| <= f_pow with |z_3|
				// substituted by 0 reduces to |x_1| + |y_2| <= f_pow, which the
				// canonical diagonal target Diag(e^{-iθ/2}, e^{+iθ/2}, 1) trivially
				// VIOLATES (|x_1| ≈ |y_2| ≈ f_pow → sum ≈ 2 f_pow), so the
				// pre-filter dropped every valid candidate and ESA() always
				// returned NO_SOLUTION.  The pre-filter was unsound — there is
				// no triangle inequality between |x_1| and |y_2| alone.  The
				// real check at line 193 with the actual z_3 is sufficient.
				y2_valid.emplace_back(y_2, c2_sq);
			}
			if(y2_valid.empty()) continue;

			for(const ringZ9& z_3 : cands[2]){
				if(interrupted) goto endthis;

				const double c_3   = abs(complex(1.0, 0.0) - z_3.toComplexDouble()/f_pow);
				const double c3_sq = c_3 * c_3;
				if(c1_sq + c3_sq > eps_sq) continue;

				const double z3_abs_sq = z_3.abs_val_sq();
				const double bound_min = min(f_pow_sq - x1_abs_sq + FPRL,
				                             f_pow_sq - z3_abs_sq + FPRL);
				const double c1c3_sq   = c1_sq + c3_sq;

				// Fix F (2026-05): same lift as for x1q.  z3z3dag was only
				// used in r2 = (rhs - z3z3dag - x3x3dag).getTerm(0).
				const int z3q = z_3.quad();
				// Fix G.1: z_3 conjugate hoisted out of fillUpper.
				const ringZ9 z3conj = z_3.complexConj();

				for(size_t x3i = 0; x3i < lookup.size(); x3i++){
					if(interrupted) goto endthis;
					const ringZ9& x_3      = lookup[x3i];
					const double x3_abs_sq = lookup_absq[x3i];
					if(c1c3_sq + x3_abs_sq / f_pow_sq > eps_sq) break;

					// Fix F: r1, r2 are now pure integer arithmetic.
					// Identity used: (x * x.complexConj()).getTerm(0) == x.quad()
					// (verified by ringZ9_unit_test invariant 14), and operator-
					// is element-wise so (a-b).getTerm(0) == a.getTerm(0)-b.getTerm(0).
					const int x3q = lookup_quad[x3i];
					const int r1  = rhs0 - x1q - x3q;
					const int r2  = rhs0 - z3q - x3q;
					if(r1 < 0 || r2 < 0) continue;

					// Per-pair cutoffs that the old per-pair pro_map enforced
					// at construction time.  Since pro_map is now global per f,
					// we apply the same gates at lookup time — same effect.
					if(r1 > bound_min || r2 > bound_min) continue;
					if(c1c3_sq + r1 / f_pow_sq >= eps_sq) continue;
					if(c1c3_sq + r2 / f_pow_sq >= eps_sq) continue;

					auto it1 = pro_map.find(r1);
					if(it1 == pro_map.end()) continue;
					const vector<ringZ9>& x2_cands = it1->second;

					auto it2 = pro_map.find(r2);
					if(it2 == pro_map.end()) continue;
					const vector<ringZ9>& y3_cands = it2->second;

					// Fix G.1: x_3 conjugate from precomputed lookup_conj table.
					const ringZ9& x3conj = lookup_conj[x3i];

					for(const auto& [y_2, c2_sq] : y2_valid){
						if(c1c3_sq + c2_sq + x3_abs_sq/f_pow_sq > eps_sq + FPRL) continue;

						if(unitaryDiagCheck(x_1, y_2, z_3, f_pow)){
							auto answer = exhaustiveCompleteUnitary(x_1, y_2, z_3, x_3, x2_cands, y3_cands,
							                                        theta, epsilon, f,
							                                        x1conj, x3conj, z3conj);
							if(answer.second){
#ifdef DEBUG
								cout << "DONE: " << f << " " << theta << " " << epsilon << endl;
#endif
								return answer.first;
							}
						}
					}
				}
			}
		}

		cands[0].clear(); cands[1].clear(); cands[2].clear();
		lookup.clear();

		// Skip f=1 and f=2: under ESA's strict filter (V exactly unitary in
		// U(3, Z[ζ_9, 1/3^f]) AND no x_i numerator divisible by 3), no V's
		// exist at those f's.
		//
		// Reason: a column entry x_i must satisfy BOTH q(x_i) = |x_i|² for
		// the per-entry q-sum / |·|²-sum constraints (Kalra Prop 2.4 + Frob
		// unitarity) to hold simultaneously — i.e., x_i must be Galois-
		// symmetric (same |·|² across all 6 embeddings of Q(ζ_9)).  The only
		// Galois-symmetric ringZ9 elements with q = 3^{2f} are c·(unit ζ_9^k)
		// where c·c̄ = 3^{2f}, i.e., c = 3^f.  All such x_i are divisible by 3
		// (since 3 = χ⁶·u in Z[ζ_9]) and rejected by !div3.
		//
		// HRSA appears to "find" V's at f=1, f=2 but those are either
		// (a) the removed buggy diagSearch path (non-unitary V's), or
		// (b) the Householder construction whose VECTOR u has small denom
		//     but whose MATRIX V = X(I − uu*) lives at sde_3 = 2·k_u, not k_u.
		// So HRSA's "f=2" Householder corresponds to ESA's f=4, not f=2.
		//
		// For ESA-vs-HRSA head-to-head, normalize on V's actual sde_3, not
		// on the algorithm-internal f label.
		f = (f == 0) ? 3 : f + 1;
	}

endthis:
	array<ringZ9chi,9> zero_matrix;
	ringZ9chi zero_el;
	for(int i = 0; i < 9; i++) zero_matrix[i] = zero_el;
#ifdef DEBUG
	cout << "FAIL" << endl;
#endif
	return zero_matrix;
}


// -----------------------------------------------------------------------
// fullDiagEnumeration  (Fix D: angle-based pruning, 2026-05)
//
// The 6-deep loop enumerates integer points (a0..a5) in the q-form ball
// [b0^2 + 3 a3^2 + b1^2 + 3 a4^2 + b2^2 + 3 a5^2] ≤ A = 4 f_pow_sq, where
// b_i = 2 a_i - a_{i+3}.  At f=3 this ball contains ~10^9 lattice points.
//
// The actual filter we want, however, is:
//   for some i ∈ {0,1,2}, Π_i(x) > eta := f_pow * (1 - eps^2/2)
// where Π_i(x) is the projection of x onto direction v_i:
//   v_0 = e^{+iθ/2},  v_1 = e^{-iθ/2},  v_2 = 1
// The candidates passing this filter live in the intersection of the q-ball
// with three (overlapping) half-planes — a small slice for small eps.
//
// Writing x = Σ_k a_k ζ_9^k (k=0..5), Π_i(x) = Σ_k a_k * mu[k][i] where
//   mu[k][i] = cos(2π k/9) * v_i.x + sin(2π k/9) * v_i.y.
// At each loop level we maintain partial[i] (sum over fixed k) and a triangle-
// inequality bound max_inner[i] on |Π_i_remaining|.  If for ALL i,
//   partial[i] + max_inner[i] ≤ eta - prune_slop,
// no completion can pass the filter — skip the whole subtree.  Pruning is
// applied at three strategic levels (after a5, after b0, after b1) where the
// remaining-budget bound on |a_k| is increasingly tight.
// -----------------------------------------------------------------------
void fullDiagEnumeration(array<vector<ringZ9>,3>& candidates, double theta, double epsilon, int f,
                         size_t max_candidates){
	const int    f_pow     = three_power(f);
	const double f_pow_d   = (double)f_pow;
	const double f_pow_eta = f_pow_d - f_pow_d * epsilon * epsilon / 2.0;
	const int    A         = 4 * f_pow * f_pow;
	const int    f_pow_sq  = f_pow * f_pow;          // Fix J: integer q-bound

	// Fix J trig constants live at file scope (JFIX_RCOS / JFIX_RSIN); used
	// inline inside the OMP parallel loop body (declared per-thread there).

	// Precompute trig constants for the three planeSide checks and the three sorts.
	// Instead of calling planeSide(test, angle) 3x — which calls toComplexDouble()
	// each time — we call toComplexDouble() once and do three dot products.
	const double cs0 = cos(-theta/2.0), sn0 = sin(-theta/2.0); // cands[0]: real(z * e^{-i*t/2})
	const double cs1 = cos( theta/2.0), sn1 = sin( theta/2.0); // cands[1]: real(z * e^{+i*t/2})
	// cands[2]: angle = 0 -> planeSide = Re(z) directly
	const complex<double> tgt0(cs1,  sn1); // sort target for cands[0]
	const complex<double> tgt1(cs1, -sn1); // sort target for cands[1]
	const complex<double> tgt2(1.0,  0.0); // sort target for cands[2]

	// Fix D: precompute mu[k][i] and |mu[k][i]| for the three filter directions.
	const double v_dir[3][2] = {
		{ cs1,  sn1},  // cands[0] target direction e^{+iθ/2}
		{ cs1, -sn1},  // cands[1] target direction e^{-iθ/2}
		{ 1.0,  0.0},  // cands[2] target direction 1
	};
	double mu[6][3], abs_mu[6][3];
	for(int k = 0; k < 6; ++k){
		const double ck = cos(2.0 * M_PI * k / 9.0);
		const double sk = sin(2.0 * M_PI * k / 9.0);
		for(int i = 0; i < 3; ++i){
			mu[k][i]     = ck * v_dir[i][0] + sk * v_dir[i][1];
			abs_mu[k][i] = std::abs(mu[k][i]);
		}
	}
	const double prune_threshold = f_pow_eta - FPRL;

	const int max_a3 = (int)ceil(sqrt(A / 3.0));

	// Fix L+M (2026-05): parallelize over (a3, a4) work units sorted by
	// descending budget_a4 (= longest-job-first scheduling).
	//
	// Why not just parallel over a3 alone (the original Fix L): the work per
	// a3 scales as ~budget_a3^3, so a3=0 dominates by orders of magnitude.
	// With dynamic scheduling, threads grab small a3 values quickly and idle
	// while one thread chews through a3=0.  At f=4 we observed CPU% drop
	// from 1721% (start) to ~941% (mid-run) due to this imbalance.
	//
	// Work-unit decomposition: for each valid (a3, a4) pair, one work unit
	// runs the inner (a5, b0, b1, b2) loops + Fix-D pruning + Fix-J filter.
	// At f=4 there are ~10^4 such units (vs only 189 for a3 alone), giving
	// fine-grained dynamic scheduling.  Sorting by descending budget_a4
	// puts the longest jobs first so they start while small jobs fill in
	// behind them — classic LPT (longest-processing-time-first) heuristic.
	struct WorkUnit { int a3, a4, budget_a4; };
	std::vector<WorkUnit> units;
	for(int a3 = -max_a3; a3 <= max_a3; ++a3){
		const int budget_a3 = A - 3*a3*a3;
		if(budget_a3 < 0) continue;
		const int max_a4 = (int)ceil(sqrt((double)budget_a3));
		for(int a4 = -max_a4; a4 <= max_a4; ++a4){
			const int budget_a4 = budget_a3 - 3*a4*a4;
			if(budget_a4 < 0) continue;
			units.push_back({a3, a4, budget_a4});
		}
	}
	std::sort(units.begin(), units.end(),
	          [](const WorkUnit& a, const WorkUnit& b){ return a.budget_a4 > b.budget_a4; });

	const int n_threads = omp_get_max_threads();
	std::vector<std::array<std::vector<ringZ9>,3>> tl_cands(n_threads);
	const int n_units = (int)units.size();

	#pragma omp parallel for schedule(dynamic, 1) default(none)               \
	    shared(tl_cands, mu, abs_mu, interrupted, JFIX_RCOS, JFIX_RSIN, units) \
	    firstprivate(n_units, prune_threshold, cs0, sn0, cs1, sn1,             \
	                 f_pow_sq, f_pow_eta, f_pow_d)
	for(int idx = 0; idx < n_units; ++idx){
		if(interrupted) continue;
		const int a3        = units[idx].a3;
		const int a4        = units[idx].a4;
		const int budget_a4 = units[idx].budget_a4;

		ringZ9 test;
		auto& cands = tl_cands[omp_get_thread_num()];
		const double* const rcos = JFIX_RCOS;
		const double* const rsin = JFIX_RSIN;

		{  // (was: outer for-loop over a4, now baked into the work unit)
			const int max_a5 = (int)ceil(sqrt((double)budget_a4));
			for(int a5 = -max_a5; a5 <= max_a5; ++a5){
				const int budget_a5 = budget_a4 - 3*a5*a5;
				if(budget_a5 < 0) continue;

				// --- Fix D pruning, level 2 (a3, a4, a5 fixed) ---
				// Free vars: a0, a1, a2.  Bound |a_k| ≤ (sqrt(budget_a5) + |a_{k+3}|)/2.
				{
					const double sqrt_bu  = std::sqrt((double)budget_a5);
					const double max_a0   = 0.5 * (sqrt_bu + std::abs(a3));
					const double max_a1   = 0.5 * (sqrt_bu + std::abs(a4));
					const double max_a2   = 0.5 * (sqrt_bu + std::abs(a5));
					bool any_reach = false;
					for(int i = 0; i < 3; ++i){
						const double partial = a3*mu[3][i] + a4*mu[4][i] + a5*mu[5][i];
						const double max_in  = max_a0*abs_mu[0][i]
						                     + max_a1*abs_mu[1][i]
						                     + max_a2*abs_mu[2][i];
						if(partial + max_in > prune_threshold){ any_reach = true; break; }
					}
					if(!any_reach) continue;
				}

				const int max_b0 = (int)ceil(sqrt((double)budget_a5));
				for(int b0 = -max_b0; b0 <= max_b0; ++b0){
					const int budget_b0 = budget_a5 - b0*b0;
					if((b0 + a3) % 2 != 0 || budget_b0 < 0) continue;
					const int a0 = (b0 + a3) / 2;

					// --- Fix D pruning, level 3 (a0 known, a1, a2 free) ---
					{
						const double sqrt_bu = std::sqrt((double)budget_b0);
						const double max_a1  = 0.5 * (sqrt_bu + std::abs(a4));
						const double max_a2  = 0.5 * (sqrt_bu + std::abs(a5));
						bool any_reach = false;
						for(int i = 0; i < 3; ++i){
							const double partial = a0*mu[0][i] + a3*mu[3][i]
							                     + a4*mu[4][i] + a5*mu[5][i];
							const double max_in  = max_a1*abs_mu[1][i]
							                     + max_a2*abs_mu[2][i];
							if(partial + max_in > prune_threshold){ any_reach = true; break; }
						}
						if(!any_reach) continue;
					}

					const int max_b1 = (int)ceil(sqrt((double)budget_b0));
					for(int b1 = -max_b1; b1 <= max_b1; ++b1){
						const int budget_b1 = budget_b0 - b1*b1;
						if((b1 + a4) % 2 != 0 || budget_b1 < 0) continue;
						const int a1 = (b1 + a4) / 2;

						// --- Fix D pruning, level 4 (a0, a1 known, only a2 free) ---
						{
							const double sqrt_bu = std::sqrt((double)budget_b1);
							const double max_a2  = 0.5 * (sqrt_bu + std::abs(a5));
							bool any_reach = false;
							for(int i = 0; i < 3; ++i){
								const double partial = a0*mu[0][i] + a1*mu[1][i]
								                     + a3*mu[3][i] + a4*mu[4][i] + a5*mu[5][i];
								const double max_in  = max_a2*abs_mu[2][i];
								if(partial + max_in > prune_threshold){ any_reach = true; break; }
							}
							if(!any_reach) continue;
						}

						const int max_b2 = (int)ceil(sqrt((double)budget_b1));
						for(int b2 = -max_b2; b2 <= max_b2; ++b2){
							if((b2 + a5) % 2 != 0) continue;
							const int a2 = (b2 + a5) / 2;

							// Fix J (2026-05): inline the filter checks so that
							// for the ~99% of (a0..a5) tuples that fail, we never
							// construct a ringZ9 — saving ctor + reduce + getter
							// calls.  ringZ9 is constructed only when at least
							// one candidate set will accept this element.

							// Inline q-form (matches ringZ9::quad), integer.
							// IMPORTANT: q(x) is the Kalra positive-definite form
							// = Tr(x · conj(x)) / 6, NOT |x|^2 for the principal
							// embedding.  See memory ringZ9_q_form.md.  So the
							// q-bound and the |x| ≤ f_pow bound are TWO DIFFERENT
							// necessary conditions — both required.
							const int q = a0*a0 + a1*a1 + a2*a2 + a3*a3 + a4*a4 + a5*a5
							            - a0*a3 - a1*a4 - a2*a5;
							if(q > f_pow_sq) continue;

							// Inline isDivisibleByInt(3): keep iff NOT all coefs
							// divisible by 3.  Note element[6..8] = 0 here, which
							// are trivially div-3, so we only check a0..a5.
							if((a0 % 3 == 0) && (a1 % 3 == 0) && (a2 % 3 == 0) &&
							   (a3 % 3 == 0) && (a4 % 3 == 0) && (a5 % 3 == 0)) continue;

							// Inline real_part / imag_part (matches ringZ9::toComplexDouble).
							const double re     = a0 + rcos[0]*a1 + rcos[1]*a2 + rcos[2]*a3 + rcos[3]*(a4+a5);
							const double im     =      rsin[0]*a1 + rsin[1]*a2 + rsin[2]*a3 + rsin[3]*(a4-a5);
							const double abs_sq = re*re + im*im;

							// Principal-embedding magnitude bound: |x| <= f_pow + FPRL,
							// equivalently |x|^2 <= (f_pow + FPRL)^2.  Replaces the
							// original abs_val() call, no sqrt.
							if(abs_sq > f_pow_sq + 2.0*f_pow_d*FPRL + FPRL*FPRL) continue;

							// Fix S (2026-05): per-Galois-conjugate magnitude bound.
							// For diagonal-entry numerators of a unitary V, the same
							// argument as in fullX3Enumeration applies: σ_k(V) is also
							// unitary, so |σ_k(numerator)|² ≤ f_pow² for k ∈ {2,4}
							// (k=1 already checked above).  Constants reuse rcos/rsin.
							const double re2 = a0 + rcos[1]*a1 + rcos[3]*a2 + rcos[2]*a3 + rcos[0]*(a4+a5);
							const double im2 =      rsin[1]*a1 + rsin[3]*a2 - rsin[2]*a3 + rsin[0]*(a5-a4);
							if(re2*re2 + im2*im2 > f_pow_sq + 2.0*f_pow_d*FPRL + FPRL*FPRL) continue;
							const double re4 = a0 + rcos[3]*a1 + rcos[0]*a2 + rcos[2]*a3 + rcos[1]*(a4+a5);
							const double im4 =      rsin[3]*a1 - rsin[0]*a2 + rsin[2]*a3 + rsin[1]*(a5-a4);
							if(re4*re4 + im4*im4 > f_pow_sq + 2.0*f_pow_d*FPRL + FPRL*FPRL) continue;

							const double eta = f_pow_eta - FPRL;
							const bool in0   = (re*cs0 - im*sn0 > eta);
							const bool in1   = (re*cs1 - im*sn1 > eta);
							const bool in2   = (re              > eta);
							if(!in0 && !in1 && !in2) continue;

							// At least one candidate set accepts; build the ringZ9 now.
							int arr[9] = {a0, a1, a2, a3, a4, a5, 0, 0, 0};
							test = ringZ9(arr);
							if(in0) cands[0].push_back(test);
							if(in1) cands[1].push_back(test);
							if(in2) cands[2].push_back(test);
						}
					}
				}
			}
		}
	}  // end parallel for over a3

	// Fix L: merge thread-local accumulators into the shared candidates[].
	for(int t = 0; t < n_threads; ++t){
		for(int i = 0; i < 3; ++i){
			candidates[i].insert(candidates[i].end(),
			                     tl_cands[t][i].begin(), tl_cands[t][i].end());
		}
	}

	// Sort by distance to each target. Precompute distances once so
	// toComplexDouble() is called O(n) times instead of O(n log n) times.
	auto sortByDist = [](vector<ringZ9>& v, complex<double> tgt){
		vector<pair<double,size_t>> d(v.size());
		for(size_t i = 0; i < v.size(); i++)
			d[i] = {abs(tgt - v[i].toComplexDouble()), i};
		sort(d.begin(), d.end());
		vector<ringZ9> tmp; tmp.reserve(v.size());
		for(auto& p : d) tmp.push_back(v[p.second]);
		v = move(tmp);
	};
	sortByDist(candidates[0], tgt0);
	sortByDist(candidates[1], tgt1);
	sortByDist(candidates[2], tgt2);

	// Keep only the top max_candidates entries (closest to each target).
	// The best candidates appear first after sorting, so truncating preserves quality
	// while bounding the downstream triple-loop cost.
	if(max_candidates > 0){
		for(int i = 0; i < 3; i++)
			if(candidates[i].size() > max_candidates)
				candidates[i].resize(max_candidates);
	}
}


double planeSide(ringZ9 test, double angle){
	return real(test.toComplexDouble() * complex<double>(cos(angle), sin(angle)));
}


int findMinQ(vector<ringZ9>& vec, int f){
	int current_min = three_power(2*f);
	for(const ringZ9& x : vec){
		int q = x.quad();
		if(q < current_min) current_min = q;
	}
	return current_min;
}


// -----------------------------------------------------------------------
// fullX3Enumeration
// -----------------------------------------------------------------------
//
// Loop-bound tightening (Fix A', 2026-05): the inner filter at line ~363
// only keeps elements with abs_val_sq <= f_pow_sq * eps_sq, AND quad ≤ f_pow_sq.
// The form-sum [b0^2 + 3 a3^2 + b1^2 + 3 a4^2 + b2^2 + 3 a5^2] = 4 q, so we can
// pick A based on the tighter of the two bounds.  q is the Kalra positive-
// definite form (= Tr(x · conj(x))/6 over Galois embeddings) — it is NOT
// |x|^2 for the principal embedding (see memory ringZ9_q_form.md), but on the
// 6-dim ringZ9 lattice the two forms are within a constant factor of each
// other so the loop bound is still a valid (slightly loose) outer bound.
//
// The original loop bound A = 4 * (f_pow_sq - M) is the *unitarity* bound.
// At small epsilon the eps-bound is far tighter (e.g. ~100x at eps=0.1, f=3),
// and elements outside the eps-bound are filtered away anyway.  Taking the
// min of the two bounds shrinks the loop region by ~1/eps^6 in the small-eps
// regime.
//
// Fix J (2026-05): inline the per-iteration filter checks (q-bound, |x|^2,
// divisibility) from the loop variables a0..a5 directly, so we never construct
// a ringZ9 for the ~99% of (a0..a5) tuples that fail.  ringZ9 is built only
// for accepted elements about to be pushed into `candidates`.
// -----------------------------------------------------------------------
void fullX3Enumeration(vector<ringZ9>& candidates, int f, int M, double epsilon){
	const double f_pow_sq = pow(3, 2*f);
	const double eps_sq   = epsilon * epsilon;
	// Fix P (2026-05): use ONLY the unitarity bound for the outer loop.  Earlier
	// "Fix A'" tightened A by min(A_unitarity, A_epsilon) under the assumption
	// that 4·quad ≈ 4·|x|^2 (principal).  This is WRONG: quad is the trace
	// (Tr(x·x̄)/6) which averages 6 Galois embeddings.  For off-diagonal entries
	// the principal embedding can have small |x|^2 (passing the eps bound) while
	// quad is much larger (because the conjugate embeddings dominate).  Fix A'
	// silently dropped every such off-diagonal — verified by esa_v_check vs the
	// HRSA V at (theta=0.5, eps=0.05, f=4): V[0,1], V[0,2], V[1,2] had |x|^2 ≈ 1
	// but quad ≈ 1400-2200, so 4·quad ≈ 6000-9000 while A_epsilon ≈ 66.
	const int A = 4 * (three_power(2*f) - M);

	const int max_a3 = (int)ceil(sqrt(A / 3.0));

	// Fix L+M (2026-05): build (a3, a4) work units sorted by descending
	// budget_a4 (LPT scheduling), then OMP-parallel-for over the unit list.
	// See fullDiagEnumeration above for full rationale.
	struct WorkUnit { int a3, a4, budget_a4; };
	std::vector<WorkUnit> units;
	for(int a3 = -max_a3; a3 <= max_a3; ++a3){
		const int budget_a3 = A - 3*a3*a3;
		if(budget_a3 < 0) continue;
		const int max_a4 = (int)ceil(sqrt((double)budget_a3));
		for(int a4 = -max_a4; a4 <= max_a4; ++a4){
			const int budget_a4 = budget_a3 - 3*a4*a4;
			if(budget_a4 < 0) continue;
			units.push_back({a3, a4, budget_a4});
		}
	}
	std::sort(units.begin(), units.end(),
	          [](const WorkUnit& a, const WorkUnit& b){ return a.budget_a4 > b.budget_a4; });

	const int n_threads = omp_get_max_threads();
	std::vector<std::vector<ringZ9>> tl_cands(n_threads);
	const int n_units = (int)units.size();

	#pragma omp parallel for schedule(dynamic, 1) default(none)               \
	    shared(tl_cands, interrupted, JFIX_RCOS, JFIX_RSIN, units)             \
	    firstprivate(n_units, f_pow_sq, eps_sq)
	for(int idx = 0; idx < n_units; ++idx){
		if(interrupted) continue;
		const int a3        = units[idx].a3;
		const int a4        = units[idx].a4;
		const int budget_a4 = units[idx].budget_a4;
		auto& my_cands = tl_cands[omp_get_thread_num()];

		{  // (was: outer for-loop over a4, now baked into the work unit)
			const int max_a5 = (int)ceil(sqrt((double)budget_a4));
			for(int a5 = -max_a5; a5 <= max_a5; ++a5){
				const int budget_a5 = budget_a4 - 3*a5*a5;
				if(budget_a5 < 0) continue;

				// Fix R (2026-05): Fix-D-style triangle-inequality lower bound
				// on principal |x|², applied at the (a3,a4,a5) level.  The
				// inner filter requires |x|² ≤ f_pow²·ε²; if the LOWEST
				// possible |x|² over (a0,a1,a2) completions already exceeds
				// that, skip the (b0,b1,b2) subtree.  Bounds on |a_i| (i=0,1,2)
				// from the budget: a_i = (b_i + a_{i+3})/2 with |b_i| ≤ √budget_a5.
				// Triangle inequality:
				//   |re| ≥ max(0, |re_partial| − max|re_remaining|)
				//   |im| ≥ max(0, |im_partial| − max|im_remaining|)
				// where partials use a3,a4,a5 only and remaining covers a0,a1,a2.
				{
					const double sqrt_bu = std::sqrt((double)budget_a5);
					const double max_a0  = 0.5 * (sqrt_bu + std::abs(a3));
					const double max_a1  = 0.5 * (sqrt_bu + std::abs(a4));
					const double max_a2  = 0.5 * (sqrt_bu + std::abs(a5));
					const double re_partial = JFIX_RCOS[2]*a3 + JFIX_RCOS[3]*(a4 + a5);
					const double im_partial = JFIX_RSIN[2]*a3 + JFIX_RSIN[3]*(a4 - a5);
					const double max_re_rem = max_a0
					                        + std::abs(JFIX_RCOS[0])*max_a1
					                        + std::abs(JFIX_RCOS[1])*max_a2;
					const double max_im_rem = std::abs(JFIX_RSIN[0])*max_a1
					                        + std::abs(JFIX_RSIN[1])*max_a2;
					const double re_lo = std::max(0.0, std::abs(re_partial) - max_re_rem);
					const double im_lo = std::max(0.0, std::abs(im_partial) - max_im_rem);
					if(re_lo*re_lo + im_lo*im_lo > f_pow_sq * eps_sq + FPRL) continue;
				}

				const int max_b0 = (int)ceil(sqrt((double)budget_a5));
				for(int b0 = -max_b0; b0 <= max_b0; ++b0){
					const int budget_b0 = budget_a5 - b0*b0;
					if((b0 + a3) % 2 != 0 || budget_b0 < 0) continue;
					const int a0 = (b0 + a3) / 2;

					// Fix R level 2 (2026-05): a0 known; a1, a2 still free.
					// Tighter principal-|x|² lower bound now that a0 is fixed.
					{
						const double sqrt_bu = std::sqrt((double)budget_b0);
						const double max_a1  = 0.5 * (sqrt_bu + std::abs(a4));
						const double max_a2  = 0.5 * (sqrt_bu + std::abs(a5));
						const double re_partial = a0
						                        + JFIX_RCOS[2]*a3
						                        + JFIX_RCOS[3]*(a4 + a5);
						const double im_partial = JFIX_RSIN[2]*a3
						                        + JFIX_RSIN[3]*(a4 - a5);
						const double max_re_rem = std::abs(JFIX_RCOS[0])*max_a1
						                        + std::abs(JFIX_RCOS[1])*max_a2;
						const double max_im_rem = std::abs(JFIX_RSIN[0])*max_a1
						                        + std::abs(JFIX_RSIN[1])*max_a2;
						const double re_lo = std::max(0.0, std::abs(re_partial) - max_re_rem);
						const double im_lo = std::max(0.0, std::abs(im_partial) - max_im_rem);
						if(re_lo*re_lo + im_lo*im_lo > f_pow_sq * eps_sq + FPRL) continue;
					}

					const int max_b1 = (int)ceil(sqrt((double)budget_b0));
					for(int b1 = -max_b1; b1 <= max_b1; ++b1){
						const int budget_b1 = budget_b0 - b1*b1;
						if((b1 + a4) % 2 != 0 || budget_b1 < 0) continue;
						const int a1 = (b1 + a4) / 2;

						// Fix R level 3 (2026-05): a0, a1 known; only a2 free.
						{
							const double sqrt_bu = std::sqrt((double)budget_b1);
							const double max_a2  = 0.5 * (sqrt_bu + std::abs(a5));
							const double re_partial = a0
							                        + JFIX_RCOS[0]*a1
							                        + JFIX_RCOS[2]*a3
							                        + JFIX_RCOS[3]*(a4 + a5);
							const double im_partial = JFIX_RSIN[0]*a1
							                        + JFIX_RSIN[2]*a3
							                        + JFIX_RSIN[3]*(a4 - a5);
							const double max_re_rem = std::abs(JFIX_RCOS[1])*max_a2;
							const double max_im_rem = std::abs(JFIX_RSIN[1])*max_a2;
							const double re_lo = std::max(0.0, std::abs(re_partial) - max_re_rem);
							const double im_lo = std::max(0.0, std::abs(im_partial) - max_im_rem);
							if(re_lo*re_lo + im_lo*im_lo > f_pow_sq * eps_sq + FPRL) continue;
						}

						const int max_b2 = (int)ceil(sqrt((double)budget_b1));
						for(int b2 = -max_b2; b2 <= max_b2; ++b2){
							if((b2 + a5) % 2 != 0) continue;
							const int a2 = (b2 + a5) / 2;

							// Fix J inline filter: skip ringZ9 ctor when any
							// of the three filters fails.

							// |x|^2 bound (the TIGHTER of the two; eps² < 1).
							const double re = a0 + JFIX_RCOS[0]*a1 + JFIX_RCOS[1]*a2 + JFIX_RCOS[2]*a3 + JFIX_RCOS[3]*(a4+a5);
							const double im =      JFIX_RSIN[0]*a1 + JFIX_RSIN[1]*a2 + JFIX_RSIN[2]*a3 + JFIX_RSIN[3]*(a4-a5);
							const double abs_sq = re*re + im*im;
							if(abs_sq > f_pow_sq * eps_sq + FPRL) continue;

							// Inline quad (Kalra q-form, integer).
							const int q = a0*a0 + a1*a1 + a2*a2 + a3*a3 + a4*a4 + a5*a5
							            - a0*a3 - a1*a4 - a2*a5;
							if((double)q > f_pow_sq + FPRL) continue;

							// Fix S (2026-05): per-Galois-conjugate magnitude bound.
							// For x = entry-numerator of a unitary V, σ_k(V) is also
							// unitary (Galois group (Z/9)* is abelian → commutes with
							// conjugation), so |σ_k(numerator)|² ≤ f_pow² for every k.
							// Pairs (1,8), (2,7), (4,5) are complex conjugates with equal
							// magnitude, so only k ∈ {1,2,4} are independent.  k=1 already
							// checked via abs_sq above (with the tighter ε² bound).  We add
							// k=2 and k=4 here at the unitarity bound f_pow².  The constants
							// reuse JFIX_RCOS/RSIN with permuted indices/signs (see Z[ζ_9]
							// canonical-basis derivation).
							const double re2 = a0 + JFIX_RCOS[1]*a1 + JFIX_RCOS[3]*a2 + JFIX_RCOS[2]*a3 + JFIX_RCOS[0]*(a4+a5);
							const double im2 =      JFIX_RSIN[1]*a1 + JFIX_RSIN[3]*a2 - JFIX_RSIN[2]*a3 + JFIX_RSIN[0]*(a5-a4);
							if(re2*re2 + im2*im2 > f_pow_sq + FPRL) continue;
							const double re4 = a0 + JFIX_RCOS[3]*a1 + JFIX_RCOS[0]*a2 + JFIX_RCOS[2]*a3 + JFIX_RCOS[1]*(a4+a5);
							const double im4 =      JFIX_RSIN[3]*a1 - JFIX_RSIN[0]*a2 + JFIX_RSIN[2]*a3 + JFIX_RSIN[1]*(a5-a4);
							if(re4*re4 + im4*im4 > f_pow_sq + FPRL) continue;

							// Inline (!isDivisibleByInt(3) || isZero):
							// reject only when all coefs divisible by 3 AND not all zero.
							const bool all_div3 = (a0%3==0) && (a1%3==0) && (a2%3==0)
							                   && (a3%3==0) && (a4%3==0) && (a5%3==0);
							const bool all_zero = (a0|a1|a2|a3|a4|a5) == 0;
							if(all_div3 && !all_zero) continue;

							// Build ringZ9 only for accepted elements.
							int arr[9] = {a0, a1, a2, a3, a4, a5, 0, 0, 0};
							my_cands.emplace_back(arr);
						}
					}
				}
			}
		}
	}

	// Fix L: merge thread-local accumulators.
	for(int t = 0; t < n_threads; ++t)
		candidates.insert(candidates.end(), tl_cands[t].begin(), tl_cands[t].end());

	sort(candidates.begin(), candidates.end(), [](const ringZ9& a, const ringZ9& b){
		return a.abs_val_sq() < b.abs_val_sq();
	});
}


bool unitaryDiagCheck(ringZ9 x_1, ringZ9 y_2, ringZ9 z_3, double f_pow){
	return (x_1.abs_val() + y_2.abs_val() - z_3.abs_val() <= f_pow + FPRL &&
	        x_1.abs_val() - y_2.abs_val() + z_3.abs_val() <= f_pow + FPRL &&
	       -x_1.abs_val() + y_2.abs_val() + z_3.abs_val() <= f_pow + FPRL);
}


pair<array<ringZ9chi,9>,bool> exhaustiveCompleteUnitary(
		ringZ9 x_1, ringZ9 y_2, ringZ9 z_3, ringZ9 x_3,
		const vector<ringZ9>& x2_cands, const vector<ringZ9>& y3_cands,
		double theta, double epsilon, int f,
		const ringZ9& x1conj, const ringZ9& x3conj, const ringZ9& z3conj){

	// Fix G.2 (2026-05): inline the per-(x_2, y_3) work that was in fillUpper
	// and precompute everything that splits as (per-x_2 term) + (per-y_3 term).
	// The original fillUpper did 6 ringZ9 multiplies per (x_2, y_3) pair, of
	// which 5 decompose into per-x_2 and per-y_3 halves:
	//   y1_num = (x2.conj * y_2)  +  (x3conj * y_3)
	//   z2_num = (x2 * x3conj)    +  (y_2 * y3.conj)
	//   z1_num = (x_1 * x3conj)   +  (y_1.first * y3.conj)
	//                ^ per (x_3)        ^ truly per-pair
	// Precompute the per-x_2 and per-y_3 quantities once each, then the inner
	// double loop is just 2 ring additions + 1 ring multiply + 3 solveSystem
	// per (x_2, y_3) pair (plus isUnitary and checkEpsilon).  Old per-pair work
	// was 6 multiplies + 4 sums + 2 conj + 3 solveSystem.
	//
	// The fillUpper function is left in place (with the post-G.1 signature) for
	// any external caller / future use, but exhaustiveCompleteUnitary no longer
	// calls it — the work is inlined here so the precomputation has access to
	// it.

	const size_t N = x2_cands.size();
	const size_t M = y3_cands.size();

	// Per-(x_3) — invariant across the (x_2, y_3) loops below.
	const ringZ9 x1_x3c = x_1 * x3conj;

	// Per-x_2 precomputations.
	vector<ringZ9> x2_y2c(N);   // x_2.conj * y_2          (used in y1_num)
	vector<ringZ9> x2_x3c(N);   // x_2      * x3conj       (used in z2_num)
	for(size_t i = 0; i < N; ++i){
		x2_y2c[i] = x2_cands[i].complexConj() * y_2;
		x2_x3c[i] = x2_cands[i] * x3conj;
	}

	// Per-y_3 precomputations.
	vector<ringZ9> x3c_y3(M);   // x3conj   * y_3          (used in y1_num)
	vector<ringZ9> y2_y3c(M);   // y_2      * y_3.conj     (used in z2_num)
	vector<ringZ9> y3_conj(M);  // y_3.conj                (used in z1_num)
	for(size_t j = 0; j < M; ++j){
		y3_conj[j] = y3_cands[j].complexConj();
		x3c_y3[j]  = x3conj * y3_cands[j];
		y2_y3c[j]  = y_2 * y3_conj[j];
	}

	// Fix O (2026-05): algebraic prefilter for the y_1 and z_2 solveSystem calls.
	// solveSystem(numer, denom) succeeds iff (numer * pn(denom)) is componentwise
	// divisible by fn(denom).  Since both numerators are sums of an i-half and a
	// j-half, distributivity lets us hoist the (* pn) multiplication out of the
	// (i,j) loop entirely:
	//   y_1: numer = -(x2_y2c[i] + x3c_y3[j]),  denom = x1conj
	//   z_2: numer = -(x2_x3c[i] + y2_y3c[j]),  denom = z3conj
	// Precompute A[i] = (x_term_i * pn) and B[j] = (y_term_j * pn) once each.
	// Per (i,j) pair, the y_1 / z_2 prechecks become 1 ringZ9 add + 1
	// isDivisibleByInt — vs the old 1 full ringZ9 mult + check.  On the (rare)
	// success path, y_1.first / z_2.first come from a cheap negate + scalar
	// divide.  z_1's numerator depends on y_1.first, so it still uses solveSystem.
	//
	// Threshold (Fix O.1): the precompute costs 2(N+M) ringZ9 mults and 4 small
	// vector allocations.  For tiny N, M this exceeds the inner-loop savings
	// (especially with 20-thread allocator contention).  Break-even analysis:
	// per-pair saving ≈ 24 int-mult-equivalents; precompute cost ≈ 72(N+M);
	// crossover at N*M ≈ 3(N+M).  Use threshold N*M >= 64 to stay clear of the
	// noise floor.  Below threshold, fall through to the original direct path.
	const bool use_prefilter = (N * M >= 64);

	const int    fn_y1 = use_prefilter ? x1conj.fieldNorm() : 0;
	const ringZ9 pn_y1 = use_prefilter ? x1conj.partialFieldNorm() : ringZ9();
	const int    fn_z2 = use_prefilter ? z3conj.fieldNorm() : 0;
	const ringZ9 pn_z2 = use_prefilter ? z3conj.partialFieldNorm() : ringZ9();

	vector<ringZ9> A_y1, A_z2, B_y1, B_z2;
	if(use_prefilter){
		A_y1.resize(N); A_z2.resize(N);
		B_y1.resize(M); B_z2.resize(M);
		for(size_t i = 0; i < N; ++i){
			A_y1[i] = x2_y2c[i] * pn_y1;
			A_z2[i] = x2_x3c[i] * pn_z2;
		}
		for(size_t j = 0; j < M; ++j){
			B_y1[j] = x3c_y3[j] * pn_y1;
			B_z2[j] = y2_y3c[j] * pn_z2;
		}
	}

	pair<array<ringZ9chi,9>,bool> answer;
	ringZ9chi zero;

	for(size_t i = 0; i < N; ++i){
		const ringZ9& x_2 = x2_cands[i];
		for(size_t j = 0; j < M; ++j){
			const ringZ9& y_3 = y3_cands[j];
			DIAG_INC(eCU_pairs);

			// y_1 — Fix O prefilter when worthwhile, otherwise direct solveSystem.
			DIAG_INC(y1_calls);
			ringZ9 y_1_first;
			if(use_prefilter){
				ringZ9 sum_y1 = A_y1[i] + B_y1[j];
				if(!sum_y1.isDivisibleByInt(fn_y1)) continue;
				y_1_first = (sum_y1 * -1) / fn_y1;
			} else {
				auto y_1 = solveSystem((x2_y2c[i] + x3c_y3[j]) * -1, x1conj);
				if(!y_1.second) continue;
				y_1_first = y_1.first;
			}
			DIAG_INC(y1_pass);

			DIAG_INC(z2_calls);
			ringZ9 z_2_first;
			if(use_prefilter){
				ringZ9 sum_z2 = A_z2[i] + B_z2[j];
				if(!sum_z2.isDivisibleByInt(fn_z2)) continue;
				z_2_first = (sum_z2 * -1) / fn_z2;
			} else {
				auto z_2 = solveSystem((x2_x3c[i] + y2_y3c[j]) * -1, z3conj);
				if(!z_2.second) continue;
				z_2_first = z_2.first;
			}
			DIAG_INC(z2_pass);

			// z_1 numerator depends on y_1.first, so it still goes through
			// solveSystem (and is reached only on the very rare double-pass).
			DIAG_INC(z1_calls);
			auto z_1 = solveSystem((x1_x3c + y_1_first * y3_conj[j]) * -1, z3conj);
			if(!z_1.second) continue;
			DIAG_INC(z1_pass);

			array<ringZ9chi,9> result;
			for(int k = 0; k < 9; ++k) result[k] = zero;
			result[0] = ringZ9chi(x_1,        f);
			result[1] = ringZ9chi(x_2,        f);
			result[2] = ringZ9chi(x_3,        f);
			result[3] = ringZ9chi(y_1_first,  f);
			result[4] = ringZ9chi(y_2,        f);
			result[5] = ringZ9chi(y_3,        f);
			result[6] = ringZ9chi(z_1.first,  f);
			result[7] = ringZ9chi(z_2_first,  f);
			result[8] = ringZ9chi(z_3,        f);

			DIAG_INC(isUnit_calls);
			if(isUnitary(result)){
				DIAG_INC(isUnit_pass);
				DIAG_INC(chkEps_calls);
				if(checkEpsilon(result, epsilon, theta)){
					DIAG_INC(chkEps_pass);
					answer.first  = result;
					answer.second = true;
					return answer;
				}
			}
		}
	}

	answer.second = false;
	return answer;
}


pair<array<ringZ9chi,9>,bool> fillUpper(array<ringZ9,9> unitary, ringZ9 x2_cand, ringZ9 y3_cand, int f,
                                        const ringZ9& x1conj, const ringZ9& x3conj, const ringZ9& z3conj){
	array<ringZ9chi,9> result;
	ringZ9chi zero;
	for(int i = 0; i < 9; i++) result[i] = zero;

	// Fix G.1 (2026-05): the conjugates of x_1 (= unitary[0]), x_3 (= unitary[2]),
	// z_3 (= unitary[8]) are now passed in pre-computed by the caller.  Inside
	// the inner-most ESA triple loop these are loop-invariant across many
	// (x_2, y_3) candidate pairs, so caller-side caching avoids billions of
	// redundant .complexConj() calls.
	//
	// y_3 conjugate is used twice in this function (z_2 and z_1) so we still
	// cache it here (Fix G.0 leftover).  x_2 conjugate is used once.
	const ringZ9 y3c = y3_cand.complexConj();      // y_3 conj (used 2x)

	auto y_1 = solveSystem((x2_cand.complexConj()*unitary[4] + x3conj*y3_cand)*-1, x1conj);
	if(!y_1.second) return {result, false};

	auto z_2 = solveSystem((x2_cand*x3conj + unitary[4]*y3c)*-1, z3conj);
	if(!z_2.second) return {result, false};

	auto z_1 = solveSystem((unitary[0]*x3conj + y_1.first*y3c)*-1, z3conj);
	if(!z_1.second) return {result, false};

	result[0] = ringZ9chi(unitary[0],   f);
	result[1] = ringZ9chi(x2_cand,      f);
	result[2] = ringZ9chi(unitary[2],   f);
	result[3] = ringZ9chi(y_1.first,    f);
	result[4] = ringZ9chi(unitary[4],   f);
	result[5] = ringZ9chi(y3_cand,      f);
	result[6] = ringZ9chi(z_1.first,    f);
	result[7] = ringZ9chi(z_2.first,    f);
	result[8] = ringZ9chi(unitary[8],   f);

	return {result, true};
}


pair<ringZ9,bool> solveSystem(ringZ9 numer, ringZ9 denom){
	// fieldNorm and partialFieldNorm are expensive (5+ ring multiplications each).
	// The denominator is always one of a small set of candidates per search level,
	// so memoizing gives near-100% hit rate after the first call per unique denom.
	const auto key = denom.getStdArray();

	int fn;
	auto fn_it = s_fieldNormCache.find(key);
	if(fn_it != s_fieldNormCache.end()){
		fn = fn_it->second;
	} else {
		fn = denom.fieldNorm();
		s_fieldNormCache.emplace(key, fn);
	}

	ringZ9 pn;
	auto pn_it = s_partialNormCache.find(key);
	if(pn_it != s_partialNormCache.end()){
		pn = pn_it->second;
	} else {
		pn = denom.partialFieldNorm();
		s_partialNormCache.emplace(key, pn);
	}

	ringZ9 new_numer = numer * pn;
	if(new_numer.isDivisibleByInt(fn)){
		return {new_numer / fn, true};
	}
	return {numer, false};
}


bool isUnitary(array<ringZ9chi,9> mat){
	ringZ9chi zero;
	ringZ9chi one(ringZ9(1), 0);

	if(mat[6]*mat[6].complexConj() + mat[7]*mat[7].complexConj() + mat[8]*mat[8].complexConj() != one ||
	   mat[3]*mat[3].complexConj() + mat[4]*mat[4].complexConj() + mat[5]*mat[5].complexConj() != one ||
	   mat[0]*mat[6].complexConj() + mat[1]*mat[7].complexConj() + mat[2]*mat[8].complexConj() != zero ||
	   mat[3]*mat[6].complexConj() + mat[4]*mat[7].complexConj() + mat[5]*mat[8].complexConj() != zero){
#ifdef DEBUG
		cout << "Matrix not a unitary." << endl;
#endif
		return false;
	}
#ifdef DEBUG
	cout << "Matrix IS unitary." << endl;
#endif
	return true;
}


bool checkEpsilon(array<ringZ9chi,9> unitary, double epsilon, double theta){
	complex<double> c_1 = complex(cos(theta/2),  sin(theta/2))  - unitary[0].toComplexDouble();
	complex<double> c_2 = complex(cos(theta/2), -sin(theta/2))  - unitary[4].toComplexDouble();
	complex<double> c_3 = complex(1.0, 0.0)                     - unitary[8].toComplexDouble();

	double norm_sq = abs(c_1)*abs(c_1) + abs(c_2)*abs(c_2) + abs(c_3)*abs(c_3)
	               + unitary[1].abs_val_sq() + unitary[2].abs_val_sq()
	               + unitary[3].abs_val_sq() + unitary[5].abs_val_sq()
	               + unitary[6].abs_val_sq() + unitary[7].abs_val_sq();

	bool success = sqrt(norm_sq) < epsilon;
#ifdef DEBUG
	if(success) cout << "Epsilon condition satisfied: " << sqrt(norm_sq) << endl;
	else        cout << "Failed epsilon condition: "    << sqrt(norm_sq) << endl;
#endif
	return success;
}


int three_power(int n){
	if(n < 0){ return 0; }
	int prod = 1;
	for(int i = 0; i < n; i++) prod *= 3;
	return prod;
}


void sortBySdeChi(vector<ringZ9>& unsorted, array<vector<ringZ9>,6>& sorted){
	for(const ringZ9& a : unsorted){
		int s = a.sdeChi();
		if(s >= 0 && s <= 5) sorted[s].push_back(a);
#ifdef DEBUG
		else cout << "sortBySdeChi: unexpected sde=" << s << endl;
#endif
	}
}


// -----------------------------------------------------------------------
// ESAWithSorting  — parallelised over the x_1 loop with OpenMP.
//
// Threading model:
//   - Enumeration, sorting, and abs_val_sq precomputation run serially.
//   - For each sde bucket k, the x_1 loop is distributed across threads with
//     dynamic scheduling (work items have very unequal cost).
//   - Each thread owns its own pro_lookup / y2_valid / x2_cands / y3_cands.
//   - The fieldNorm/partialFieldNorm memos are thread_local, so each thread
//     builds its own cache independently — no contention.
//   - A shared atomic<bool> `found` + mutex-protected result array allow the
//     first thread to find a solution to signal the others to stop.
//   - Ctrl-C (interrupted) is checked inside the parallel loop and triggers
//     cancellation via the found flag path.
// -----------------------------------------------------------------------
array<ringZ9chi, 9> ESAWithSorting(double theta, double epsilon, int max_f){
	// See note in ESA(): negate theta to match the unified paper convention
	// (target Diag(e^{-i theta/2}, e^{+i theta/2}, 1)) without rewriting the
	// internal sin/cos signs.
	theta = -theta;

	int f = 4;
	array<vector<ringZ9>,3> cands;
	vector<ringZ9> lookup;
	array<array<vector<ringZ9>,6>,3> sorted_cands;
	array<vector<ringZ9>,6> sorted_lookup;
	vector<ringZ9> blank;

	// Shared result state for the parallel search
	array<ringZ9chi,9> shared_result;
	std::atomic<bool> found(false);
	omp_lock_t result_lock;
	omp_init_lock(&result_lock);

	while(f <= max_f && !found){
		const double f_pow      = pow(3, f);
		const double f_pow_sq   = pow(3, 2*f);
		const int    f_pow_sq_i = three_power(2*f);
		ringZ9 rhs(f_pow_sq_i);

#ifdef DEBUG
		cout << "f: " << f << endl;
#endif

		// Fix P (2026-05): max_candidates=0 → keep ALL diag candidates (no
		// top-500 truncation).  The truncation silently dropped HRSA's V's
		// diagonals at (theta=0.5, eps=0.05, f=4) — verified by esa_v_check.
		fullDiagEnumeration(cands, theta, epsilon, f, /*max_candidates=*/0);

		int minQ = min(findMinQ(cands[0], f), findMinQ(cands[2], f));
		fullX3Enumeration(lookup, f, minQ, epsilon);

		sortBySdeChi(cands[0], sorted_cands[0]); cands[0].clear();
		sortBySdeChi(cands[1], sorted_cands[1]); cands[1].clear();
		sortBySdeChi(cands[2], sorted_cands[2]); cands[2].clear();
		sortBySdeChi(lookup,   sorted_lookup);    lookup.clear();

		for(int i = 0; i < 6; i++) sorted_lookup[i].push_back(ringZ9(0));

		// Precompute abs_val_sq for each sde bucket (read-only in parallel section)
		array<vector<double>,6> sl_absq;
		array<vector<int>,6>    sl_quad;   // Fix F: integer quad() per lookup entry
		array<vector<ringZ9>,6> sl_conj;   // Fix G.1: complexConj per lookup entry
		for(int i = 0; i < 6; i++){
			sl_absq[i].resize(sorted_lookup[i].size());
			sl_quad[i].resize(sorted_lookup[i].size());
			sl_conj[i].resize(sorted_lookup[i].size());
			for(size_t j = 0; j < sorted_lookup[i].size(); j++){
				sl_absq[i][j] = sorted_lookup[i][j].abs_val_sq();
				sl_quad[i][j] = sorted_lookup[i][j].quad();
				sl_conj[i][j] = sorted_lookup[i][j].complexConj();
			}
		}

		// --------------------------------------------------------------------
		//  Fix B (2026-05): hoist pro_map out of the per-thread inner loop.
		//  Build one shared, read-only hash map per sde bucket BEFORE the
		//  parallel region.  Inner loop applies the per-pair bound_min /
		//  eps gates at lookup time (cheap integer compares).  Eliminates
		//  the per-(x_1, z_3) thread-local rebuild that previously dominated
		//  serial work inside each thread.
		// --------------------------------------------------------------------
		array<unordered_map<int, vector<ringZ9>>, 6> sl_pro_map;
		for(int i = 0; i < 6; i++){
			sl_pro_map[i].reserve(sorted_lookup[i].size());
			for(size_t j = 0; j < sorted_lookup[i].size(); j++){
				sl_pro_map[i][sl_quad[i][j]]
					.push_back(sorted_lookup[i][j]);
			}
		}

		// Fix F: integer constant term of rhs (rhs is constructed as
		// ringZ9(f_pow_sq_i), so rhs.getTerm(0) = f_pow_sq_i = 3^{2f}).
		const int rhs0 = rhs.getTerm(0);

		// Clear the thread-local memoisation caches once per f-level (serial section).
		// Each thread will repopulate its own cache during the parallel region.
		clearSolveSystemCache();
		const double eps_sq = epsilon * epsilon;

		// =========================================================================
		// Fix N (2026-05): work-unit decomposition for the inner triple loop.
		//
		// The previous structure was:
		//   for k = 0..5:                            # sequential
		//     #pragma omp parallel for over x_1 in x1_bucket[k]
		//
		// Two failure modes observed at f=4:
		//   (1) small buckets gave low parallelism (e.g., 4 cores when k has
		//       only a few candidates);
		//   (2) early `if(c1_sq > eps_sq) continue;` rejected most x_1's,
		//       leaving very few threads doing real work.
		//
		// Fix N: pre-build per-(k, x_1) X1Info structures (computing c_1,
		// c1_sq, x1conj, y2_valid up front and DROPPING any with c1_sq > eps
		// or empty y2_valid).  Then build (k, x_1, z_3) work units (also
		// dropping pairs with c1_sq+c3_sq > eps).  Single OMP parallel-for
		// over all valid work units across all six buckets.  Within each
		// unit, run only the x_3 inner loop (the most expensive part).
		// =========================================================================
		struct X1Info {
			int k, x1i;
			double c1_sq, x1_abs_sq;
			int x1q;
			ringZ9 x1conj;
			std::vector<std::pair<ringZ9,double>> y2_valid;
		};
		std::vector<X1Info> x1_infos;
		for(int k = 0; k < 6; ++k){
			const auto& x1_bucket = sorted_cands[0][k];
			const auto& y2_bucket = sorted_cands[1][k];
			for(size_t i = 0; i < x1_bucket.size(); ++i){
				const ringZ9& x_1 = x1_bucket[i];
				const double c_1   = abs(complex(cos(theta/2), sin(theta/2))
				                         - x_1.toComplexDouble()/f_pow);
				const double c1_sq = c_1 * c_1;
				if(c1_sq > eps_sq) continue;

				X1Info info;
				info.k = k;
				info.x1i = (int)i;
				info.c1_sq = c1_sq;
				info.x1_abs_sq = x_1.abs_val_sq();
				info.x1q = x_1.quad();
				info.x1conj = x_1.complexConj();
				info.y2_valid.reserve(y2_bucket.size());
				for(const ringZ9& y_2 : y2_bucket){
					const double c_2 = abs(complex(cos(theta/2), -sin(theta/2))
					                        - y_2.toComplexDouble()/f_pow);
					const double c2_sq = c_2 * c_2;
					if(c1_sq + c2_sq > eps_sq + FPRL) continue;
					info.y2_valid.emplace_back(y_2, c2_sq);
				}
				if(info.y2_valid.empty()) continue;
				x1_infos.push_back(std::move(info));
			}
		}

		struct WorkUnit {
			int xinfo;
			int z3i;
			double c1c3_sq, bound_min;
			int z3q;
			ringZ9 z3conj;
		};
		std::vector<WorkUnit> work_units;
		for(size_t xi = 0; xi < x1_infos.size(); ++xi){
			const X1Info& info = x1_infos[xi];
			const int k = info.k;
			const auto& z3_bucket = sorted_cands[2][k];
			for(size_t j = 0; j < z3_bucket.size(); ++j){
				const ringZ9& z_3 = z3_bucket[j];
				const double c_3   = abs(complex(1.0, 0.0) - z_3.toComplexDouble()/f_pow);
				const double c3_sq = c_3 * c_3;
				if(info.c1_sq + c3_sq > eps_sq) continue;

				WorkUnit wu;
				wu.xinfo = (int)xi;
				wu.z3i = (int)j;
				wu.c1c3_sq = info.c1_sq + c3_sq;
				const double z3_abs_sq = z_3.abs_val_sq();
				wu.bound_min = std::min(f_pow_sq - info.x1_abs_sq + FPRL,
				                        f_pow_sq - z3_abs_sq + FPRL);
				wu.z3q = z_3.quad();
				wu.z3conj = z_3.complexConj();
				work_units.push_back(std::move(wu));
			}
		}

		const int n_units = (int)work_units.size();

		std::atomic<int> done_units{0};
		auto wu_t0 = std::chrono::steady_clock::now();
		fprintf(stderr, "[f=%d ESA] starting %d work units\n", f, n_units);
		DIAG_RESET();
		DIAG_SET_F(f);

		#pragma omp parallel for schedule(dynamic,1) default(none) \
		    shared(work_units, x1_infos, sorted_cands, sorted_lookup, \
		           sl_absq, sl_quad, sl_conj, sl_pro_map, \
		           rhs, found, shared_result, result_lock, interrupted, \
		           n_units, done_units, wu_t0, stderr DIAG_OMP_SHARED) \
		    firstprivate(f_pow, f_pow_sq, eps_sq, theta, epsilon, f, rhs0)
		for(int u = 0; u < n_units; ++u){
			if(found || interrupted) continue;

			const WorkUnit& wu = work_units[u];
			const X1Info& info = x1_infos[wu.xinfo];
			const int k = info.k;
			const ringZ9& x_1  = sorted_cands[0][k][info.x1i];
			const ringZ9& z_3  = sorted_cands[2][k][wu.z3i];
			const auto&  sl_k        = sorted_lookup[k];
			const auto&  sl_absq_k   = sl_absq[k];
			const auto&  sl_quad_k   = sl_quad[k];
			const auto&  sl_conj_k   = sl_conj[k];
			const auto&  pro_map_k   = sl_pro_map[k];
			const auto&  y2_valid    = info.y2_valid;
			const int    x1q         = info.x1q;
			const ringZ9& x1conj     = info.x1conj;
			const int    z3q         = wu.z3q;
			const ringZ9& z3conj     = wu.z3conj;
			const double bound_min   = wu.bound_min;
			const double c1c3_sq     = wu.c1c3_sq;

			for(size_t x3i = 0; x3i < sl_k.size(); ++x3i){
				if(found || interrupted) break;

				DIAG_INC(x3_entries);
				const double x3_abs_sq = sl_absq_k[x3i];
				if(c1c3_sq + x3_abs_sq / f_pow_sq > eps_sq) break;

				const ringZ9& x_3 = sl_k[x3i];
				const int x3q = sl_quad_k[x3i];
				const int r1  = rhs0 - x1q - x3q;
				const int r2  = rhs0 - z3q - x3q;
				if(r1 < 0 || r2 < 0) continue;

				if(r1 > bound_min || r2 > bound_min) continue;
				if(c1c3_sq + r1 / f_pow_sq >= eps_sq) continue;
				if(c1c3_sq + r2 / f_pow_sq >= eps_sq) continue;

				auto it1 = pro_map_k.find(r1);
				if(it1 == pro_map_k.end()) continue;
				const vector<ringZ9>& x2_cands = it1->second;

				auto it2 = pro_map_k.find(r2);
				if(it2 == pro_map_k.end()) continue;
				const vector<ringZ9>& y3_cands = it2->second;

				const ringZ9& x3conj = sl_conj_k[x3i];
				DIAG_INC(x3_passed);

				for(const auto& [y_2, c2_sq] : y2_valid){
					if(found) break;
					DIAG_INC(y2_iters);
					if(c1c3_sq + c2_sq + x3_abs_sq/f_pow_sq > eps_sq + FPRL) continue;
					DIAG_INC(y2_passed);

					DIAG_INC(udc_calls);
					if(unitaryDiagCheck(x_1, y_2, z_3, f_pow)){
						DIAG_INC(udc_pass);
						DIAG_INC(eCU_calls);
						auto answer = exhaustiveCompleteUnitary(
						    x_1, y_2, z_3, x_3, x2_cands, y3_cands,
						    theta, epsilon, f,
						    x1conj, x3conj, z3conj);
						if(answer.second){
							omp_set_lock(&result_lock);
							if(!found){
								shared_result = answer.first;
								found = true;
							}
							omp_unset_lock(&result_lock);
							break;
						}
					}
				}
			}

			int dnu = ++done_units;
			if((dnu & 1023) == 0){
				double el = std::chrono::duration<double>(
					std::chrono::steady_clock::now() - wu_t0).count();
				double rate = el > 0 ? dnu / el : 0;
				double eta = rate > 0 ? (n_units - dnu) / rate : 0;
				fprintf(stderr,
					"[f=%d ESA] units %d/%d (%.1f%%) el=%.1fs rate=%.0f/s ETA=%.0fs\n",
					f, dnu, n_units, 100.0*dnu/n_units, el, rate, eta);
				DIAG_PRINT(f);
			}
		} // end omp parallel for over work units

		DIAG_PRINT(f);

		if(found) break;

		// Skip f=1 and f=2 (see detailed comment in ESA() above).
		// Briefly: at f=1, f=2 every V meeting unitarity has all entries of
		// form c·(unit ζ_9^k) with c=3^f, all divisible by 3 → rejected by
		// the !div3 filter.
		f = (f == 0) ? 3 : f + 1;
		sorted_cands[0].fill(blank);
		sorted_cands[1].fill(blank);
		sorted_cands[2].fill(blank);
		sorted_lookup.fill(blank);
	}

	omp_destroy_lock(&result_lock);

	if(found) return shared_result;

	array<ringZ9chi,9> zero_matrix;
	ringZ9chi zero_el;
	for(int i = 0; i < 9; i++) zero_matrix[i] = zero_el;
#ifdef DEBUG
	cout << "FAIL" << endl;
#endif
	return zero_matrix;
}


//**********************************//
//********* DEBUG TOOLS ************//
//**********************************//

array<ringZ9chi, 9> testDiagonalESA(ringZ9 a1, ringZ9 a2, ringZ9 a3, double theta, double epsilon, int max_f){
	// See note in ESA(): negate theta for unified-convention alignment.
	theta = -theta;

	int f = 0;
	array<vector<ringZ9>,3> cands;
	vector<ringZ9> lookup;

	while(f <= max_f){
		const double f_pow    = pow(3, f);
		const double f_pow_sq = pow(3, 2*f);
		const int    f_pow_sq_i = three_power(2*f);
		ringZ9 rhs(f_pow_sq_i);

		cands[0].push_back(a1);
		cands[1].push_back(a2);
		cands[2].push_back(a3);

		if(!foundInDiagTable(a1, a2, a3, theta, epsilon, f)){
			cout << "Couldn't find diagonal." << endl;
		}

		int minQ = min(findMinQ(cands[0], f), findMinQ(cands[2], f));
		fullX3Enumeration(lookup, f, minQ, epsilon);
		cout << "Lookup table size: " << lookup.size() << endl;

		const double eps_sq = epsilon * epsilon;
		vector<ringZ9> pro_lookup;
		pro_lookup.reserve(lookup.size());

		for(const ringZ9& x_1 : cands[0]){
			const double c_1   = abs(complex(cos(theta/2), sin(theta/2)) - x_1.toComplexDouble()/f_pow);
			const double c1_sq = c_1 * c_1;
			if(c1_sq > eps_sq) continue;

			const double x1_abs_sq = x_1.abs_val_sq();
			const ringZ9 x1x1dag   = x_1 * x_1.complexConj();

			for(const ringZ9& z_3 : cands[2]){
				const double c_3   = abs(complex(1.0, 0.0) - z_3.toComplexDouble()/f_pow);
				const double c3_sq = c_3 * c_3;
				cout << c1_sq + c3_sq << " " << eps_sq << endl;
				if(c1_sq + c3_sq > eps_sq) continue;

				const double z3_abs_sq = z_3.abs_val_sq();
				const double bound_min = min(f_pow_sq - x1_abs_sq + FPRL, f_pow_sq - z3_abs_sq + FPRL);
				const double c1c3_sq   = c1_sq + c3_sq;

				pro_lookup.clear();
				cout << "prelookup size: " << lookup.size() << endl;
				for(const ringZ9& x : lookup){
					const double abs_sq = x.abs_val_sq();
					if(abs_sq > bound_min) continue;
					if(c1c3_sq + abs_sq / f_pow_sq >= eps_sq) continue;
					pro_lookup.push_back(x);
				}
				cout << "postlookup size: " << pro_lookup.size() << endl;

				const ringZ9 z3z3dag = z_3 * z_3.complexConj();

				for(const ringZ9& x_3 : lookup){
					const double x3_abs_sq = x_3.abs_val_sq();
					if(c1c3_sq + x3_abs_sq / f_pow_sq > eps_sq) continue;

					cout << "MADE IT HERE: "; x_3.print(); cout << "\n";

					const ringZ9 x3x3dag = x_3 * x_3.complexConj();
					const ringZ9 rhs1 = rhs - x1x1dag - x3x3dag;
					const ringZ9 rhs2 = rhs - z3z3dag - x3x3dag;
					const double rhs1_val = (double)rhs1.getTerm(0);
					const double rhs2_val = (double)rhs2.getTerm(0);

					vector<ringZ9> x2_cands, y3_cands;
					for(const ringZ9& x : pro_lookup){
						if(fabs(x.abs_val_sq() - rhs1_val) < FPRL) x2_cands.push_back(x);
					}
					if(x2_cands.empty()) continue;
					for(const ringZ9& x : pro_lookup){
						if(fabs(x.abs_val_sq() - rhs2_val) < FPRL) y3_cands.push_back(x);
					}
					if(y3_cands.empty()) continue;

					// Fix G.1: hoist conjugates for the Fix-G.1 signature.
					// (testDiagonalESA is a debug routine called rarely, so we
					// just compute these inline rather than precomputing.)
					const ringZ9 x1conj = x_1.complexConj();
					const ringZ9 z3conj = z_3.complexConj();
					const ringZ9 x3conj = x_3.complexConj();

					for(const ringZ9& y_2 : cands[1]){
						const double c_2   = abs(complex(cos(theta/2), -sin(theta/2)) - y_2.toComplexDouble()/f_pow);
						if(c1c3_sq + c_2*c_2 + x3_abs_sq/f_pow_sq > eps_sq + FPRL) continue;

						if(unitaryDiagCheck(x_1, y_2, z_3, f_pow)){
							auto answer = exhaustiveCompleteUnitary(x_1, y_2, z_3, x_3, x2_cands, y3_cands, theta, epsilon, f,
							                                        x1conj, x3conj, z3conj);
							if(answer.second){
								cout << "DONE: " << f << " " << theta << " " << epsilon << endl;
								return answer.first;
							}
							cout << "Failed on iteration:\n";
							x_1.print(); cout << " "; y_2.print(); cout << " "; z_3.print(); cout << "\n";
						}
					}
				}
			}
		}

		cands[0].clear(); cands[1].clear(); cands[2].clear();
		lookup.clear();
		f++;
	}

	array<ringZ9chi,9> zero_matrix;
	ringZ9chi zero_el;
	for(int i = 0; i < 9; i++) zero_matrix[i] = zero_el;
	cout << "FAIL" << endl;
	return zero_matrix;
}


array<ringZ9chi, 9> testDiagonalESAWithSorting(ringZ9 a1, ringZ9 a2, ringZ9 a3, double theta, double epsilon, int max_f){
	// Delegate to testDiagonalESA for simplicity in debug context
	return testDiagonalESA(a1, a2, a3, theta, epsilon, max_f);
}


bool foundInDiagTable(ringZ9 x_1, ringZ9 y_2, ringZ9 z_3, double theta, double epsilon, int f){
	array<vector<ringZ9>,3> cands;
	fullDiagEnumeration(cands, theta, epsilon, f, /*max_candidates=*/0);

	bool x1_flag = false, y2_flag = false, z3_flag = false;

	for(const ringZ9& a : cands[0]){ if(a == x_1){ x1_flag = true; break; } }
	for(const ringZ9& a : cands[1]){ if(a == y_2){ y2_flag = true; break; } }
	for(const ringZ9& a : cands[2]){ if(a == z_3){ z3_flag = true; break; } }

	if(x1_flag) cout << "x_1 was found." << endl;
	if(y2_flag) cout << "y_2 was found." << endl;
	if(z3_flag) cout << "z_3 was found." << endl;

	return x1_flag && y2_flag && z3_flag;
}


void handleCtrlC(int sig){
	interrupted = true;
}
