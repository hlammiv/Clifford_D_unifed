#include "householder_search.h"
#include "decompose.h"
#include "rf_features.h"
#include <algorithm>
#include <mutex>
#include <omp.h>
#include <climits>

using namespace std;

// FILTER_REL_TOL: relative tolerance added to the candidate pre-filters in
// entryEnumeration to avoid discarding borderline candidates due to floating-point
// rounding error in the filter comparisons.
//
// The three filters each compare a computed float quantity against a threshold T:
//   x1, x3 filters:  computed_val < eps_cond          (T = eps_cond)
//   x2 filter:       computed_val < f_pow * eps_cond   (T = f_pow * eps_cond)
//
// A candidate whose true value is just barely below T might be computed as just
// barely above T due to float rounding, and incorrectly excluded. The filter buffer
// must therefore scale with T, not be a fixed constant. We widen each threshold by
// a relative factor (1 + FILTER_REL_TOL) so the buffer shrinks with eps_cond.
//
// At eps=0.1:   eps_cond~1.25e-3, buffer~1.25e-9  (was 1e-4 with old fixed FPRL)
// At eps=0.01:  eps_cond~1.25e-5, buffer~1.25e-11 (old FPRL was 8x eps_cond!)
// At eps=1e-10: eps_cond~1.25e-21,buffer~1.25e-27 (old FPRL was 8e16x eps_cond!)
//
// 1e-6 provides ~1 million times the actual float rounding error (~2e-16 relative),
// a comfortable margin with no risk of letting genuinely-invalid candidates through.
//
// FPRL must NOT appear in epsTest — that is the strict mathematical acceptance gate.
// The pre-filters are intentionally conservative (they may admit extra candidates),
// and epsTest is what enforces the actual epsilon condition.
const double FILTER_REL_TOL = 1e-6;
extern std::atomic<bool> interrupted;
extern bool g_hrsa_alt_order;  // defined in decompose.cpp; toggles outer x_1 alternating iteration
extern bool g_hrsa_mod3_filter;  // defined in decompose.cpp; Kalra-Mosca-Valluri 2023 Thm 3.7 prune
extern int g_hrsa_rf_gate;     // defined in decompose.cpp; if >0, take union(distance-top-K, RF-top-K) for outer iteration

// Global mutex protecting all cout output. Without this, concurrent epsTest calls
// from the parallel x1 loop produce interleaved/garbled output.
static mutex cout_mutex;

array<ringZ9chi,3> HRSA(double theta, double epsilon, int max_f, double c){
	int f = 0;
	ringZ9chi zero;
	array<ringZ9chi,3> answer;
	vector<ringZ9> x1_cands, x2_cands;
	// OPT: lookup is now a hash map keyed by quad() value, with each bucket a
	// contiguous vector<ringZ9>. This turns the inner x_3 scan from
	// O(|lookup|) to O(1) average plus a cache-friendly linear walk over the
	// bucket, eliminating the dominant cost of the original triple-nested loop.
	unordered_map<int,vector<ringZ9>> lookup;

	// Convention: theta is the canonical angle such that the target gate is
	// R^Z_{(0,1)}(theta) = Diag(e^{-i theta/2}, e^{+i theta/2}, 1). The Householder
	// vector is u = (1/sqrt 2)(e^{+i theta/2}, -1, 0) (paper Eq. preceding (4)).
	// We therefore search for x_1 ~ e^{+i theta/2}.
	complex<double> angle_dir(cos(theta/2.0), sin(theta/2.0));
	
	answer[0] = zero; answer[1] = zero; answer[2] = zero;
	
	while( f <= max_f){
		// OPT: use three_power() for exact integer arithmetic instead of floating-point pow().
		// static_cast<int>(pow(3,2*f)) can silently return the wrong value for larger f
		// due to floating-point rounding (e.g., 3^10 = 59049.9999... -> 59048).
		int f_pow_sq = three_power(f) * three_power(f);
		ringZ9 f_pow_sq_doub(2*f_pow_sq);
		
		cout << "Enumerating entry candidates for f = " << f << "." << endl;
		entryEnumeration(x1_cands, x2_cands, lookup, theta, epsilon, f, c);

		// Per-conjugate norm caches.  For r in {2,4}, |sigma_r(X_i)|^2 is needed
		// to enforce the per-Galois-orbit joint sum bound at triple formation.
		// Identity: sum_i |sigma_r(X_i)|^2 = 2*3^{2f} for each r in {1,2,4}
		// (rational integer is Galois-fixed).  Currently the loop only checks
		// q1+q2 <= 2*3^{2f}, which is the AVERAGE over r.  Per-r checks are
		// strictly stronger -- they prune (X_1,X_2) pairs whose joint sum at
		// some conjugate level already exceeds 2*3^{2f}, so no X_3 can complete
		// the triple.
		const int N1 = (int)x1_cands.size();
		const int N2 = (int)x2_cands.size();
		std::vector<double> x1_s2sq(N1), x1_s4sq(N1);
		std::vector<double> x2_s2sq(N2), x2_s4sq(N2);
		#pragma omp parallel for schedule(static) if(N1 + N2 >= 1024)
		for(int idx = 0; idx < N1 + N2; ++idx){
			if(idx < N1){
				x1_s2sq[idx] = x1_cands[idx].GaloisAut(2).abs_val_sq();
				x1_s4sq[idx] = x1_cands[idx].GaloisAut(4).abs_val_sq();
			} else {
				int j = idx - N1;
				x2_s2sq[j] = x2_cands[j].GaloisAut(2).abs_val_sq();
				x2_s4sq[j] = x2_cands[j].GaloisAut(4).abs_val_sq();
			}
		}
		// Tolerance: same FILTER_REL_TOL philosophy -- absorb float roundoff
		// in GaloisAut+abs_val_sq without admitting genuinely invalid pairs.
		const double conj_thr = 2.0 * (double)f_pow_sq * (1.0 + FILTER_REL_TOL);

		// Parallelise over x1_cands when f >= 2: each x1 is fully independent.
		// lookup is read-only at this point, so concurrent find() calls are safe.
		// 'found' signals the first thread to pass epsTest; others check and exit early.
		// Below f=2 the candidate lists are small and thread overhead exceeds the gain,
		// so we fall through to a simple serial loop for f=0 and f=1.
		atomic<bool> found(false);
		mutex answer_mutex;
		array<ringZ9chi,3> local_best;

		#pragma omp parallel for schedule(dynamic) if(f >= 2) \
		    shared(found, answer_mutex, local_best)
		for(int i = 0; i < (int)x1_cands.size(); ++i){

			if(found || interrupted) continue;

			const ringZ9& x_1 = x1_cands[i];
			int q1 = x_1.quad();
			const double s2_x1 = x1_s2sq[i];
			const double s4_x1 = x1_s4sq[i];

			for(int j = 0; j < (int)x2_cands.size(); ++j){

				if(found || interrupted) break;

				const ringZ9& x_2 = x2_cands[j];
				int q2 = x_2.quad();
				if(q1 + q2 > 2 * f_pow_sq) continue;
				// Per-conjugate joint sum check: pair must satisfy the per-Galois-orbit
				// budget at sigma_2 and sigma_4, not just the average (q-form sum).
				if(s2_x1 + x2_s2sq[j] > conj_thr) continue;
				if(s4_x1 + x2_s4sq[j] > conj_thr) continue;

				// The hash map gives us all x3 candidates with quad(x3) == target_norm.
				// quad(x3) is only the constant term of complexConj(x3)*x3; that product
				// is not always a rational integer. We must verify the full ring equality
				// complexConj(x1)*x1 + complexConj(x2)*x2 + complexConj(x3)*x3 == 2*3^{2f}
				// (as a ring element, not just its constant term) to guarantee |u|^2 = 2
				// exactly. The hash map is a fast pre-filter; this exact check culls false
				// positives that would produce non-unitary H matrices.
				int target_norm = 2*f_pow_sq - q1 - q2;
				ringZ9 x3sq = f_pow_sq_doub - x_1.complexConj()*x_1 - x_2.complexConj()*x_2;

				auto bucket_it = lookup.find(target_norm);
				if(bucket_it == lookup.end()) continue;
				for(const ringZ9& x_3 : bucket_it->second){

					if(found) break;

					// Exact ring check: reject x3 candidates where the ring product
					// has non-zero higher-order terms (i.e. is not a rational integer).
					if(!(x_3.complexConj()*x_3 == x3sq)) continue;

					array<ringZ9chi,3> candidate = {
						ringZ9chi(x_1,f),
						ringZ9chi(x_2,f),
						ringZ9chi(x_3,f)
					};

					// epsTest is lock-free: no cout inside, pure arithmetic.
					if(epsTest(candidate[0], candidate[1], candidate[2], f, angle_dir, epsilon, c)){
						lock_guard<mutex> lk(answer_mutex);
						if(!found){
							found = true;
							local_best = candidate;
						}
					}
				}
			}
		}

		if(found){
			answer = local_best;
			// Log the winning eps_diff now that we're back in the serial region.
			double eps_diff = pow(abs(answer[0].toComplexDouble() - angle_dir),2)
			                + (answer[1] + ringZ9chi(ringZ9(1),0)).abs_val_sq()
			                + answer[2].abs_val_sq();
			cout << "Epsilon Diff. Val.: " << eps_diff
			     << " Eps. Cond.: " << epsilon*epsilon/(8.0*c*c) << endl;
			cout << "Success!" << endl;
			return answer;
		}
		if(interrupted) return answer;
		
		f++;
		x1_cands.clear();
		x2_cands.clear();
		lookup.clear();
	}
	cout << "Failure." << endl;
	return answer;
}

void entryEnumeration(	vector<ringZ9>& x1_cands,
						vector<ringZ9>& x2_cands,
						unordered_map<int,vector<ringZ9>>& lookup,
						double theta, double epsilon, int f, double c){
	// Convention: theta is canonical (target = R^Z_{(0,1)}(theta));
	// search target for x_1 is e^{+i theta/2}.
	complex<double> target_ang(cos(theta/2.0),sin(theta/2.0));
	double target_re = real(target_ang);
	double target_im = imag(target_ang);
	int f_pow = three_power(f);
	
	// A is the budget for the integer quadratic form 4*q(x) = sum b_i^2 + 3*sum a_{i+3}^2,
	// which bounds q(x) <= A/4.  Joint-Galois identity gives Sigma_i q(X_i) = 2*3^{2f}
	// (since q is the avg of the 3 conjugate norms, and each conjugate's joint sum is 2*3^{2f}),
	// so per-coord max is q(X_i) <= 2*3^{2f}, requiring A = 8*3^{2f}.  Original v1 used
	// A = 4*3^{2f} as a heuristic shortcut (Chris's deleted comment: "To truly exhaust
	// all possible vectors, we should actually multiply by a factor of 8 instead of 4").
	int A = 8 * f_pow*f_pow;
	
	double eps_cond = epsilon*epsilon/(8.0*c*c);
	
	// OPT: precompute 1/3^f once for converting ringZ9 coefficients to complex doubles
	// without constructing and normalizing a full ringZ9chi object each time.
	double inv_3f = 1.0;
	for (int i = 0; i < f; ++i) inv_3f /= 3.0;

	// Precomputed trig tables matching ringZ9chi::real_part/imag_part static arrays.
	static const double cos_vals[6] = {
		1.0, 0.766044443118978, 0.17364817766693041,
		-0.5, -0.9396926207859083, -0.9396926207859084,
	};
	static const double sin_vals[6] = {
		0.0, 0.6427876096865393, 0.984807753012208,
		0.8660254037844387, 0.3420201433256689, -0.34202014332566866,
	};

	ringZ9 test;	
	int max_a3 = static_cast<int>(ceil(sqrt(A / 3.0)));

	// Parallelise the outermost loop over a3. Each a3 value is fully independent.
	// Threads accumulate into thread-local vectors/maps to avoid contention, then
	// merge into the shared outputs after the parallel region.
	// dynamic scheduling balances load: work per a3 varies widely since inner budgets
	// collapse quickly for large |a3|.
	vector<vector<ringZ9>> tl_x1(omp_get_max_threads());
	vector<vector<ringZ9>> tl_x2(omp_get_max_threads());
	vector<unordered_map<int,vector<ringZ9>>> tl_lookup(omp_get_max_threads());

	// 2026-05-13: emit periodic stderr progress lines so external watchers can
	// see how far enumeration has progressed (useful for tuning timeouts on
	// tight-ε cells where enumeration alone can exceed 30 min).
	int total_a3 = 2 * max_a3 + 1;
	int enum_progress = 0;
	int report_every = std::max(1, total_a3 / 20);  // ~20 progress lines total

	#pragma omp parallel for schedule(dynamic) if(f >= 2) shared(tl_x1, tl_x2, tl_lookup, enum_progress)
	for (int a3 = -max_a3; a3 <= max_a3; ++a3){

		if(interrupted) continue;

		int my_progress;
		#pragma omp atomic capture
		my_progress = ++enum_progress;
		if (my_progress % report_every == 0 || my_progress == total_a3) {
			std::cerr << "[enum] f=" << f << " a3-progress="
			          << my_progress << "/" << total_a3 << std::endl;
		}

		int tid = omp_get_thread_num();
		auto& loc_x1     = tl_x1[tid];
		auto& loc_x2     = tl_x2[tid];
		auto& loc_lookup = tl_lookup[tid];
		
		int budget_a3 = A - 3*a3*a3;
		if (budget_a3 < 0) continue;
		
		int max_a4 = static_cast<int>(ceil(sqrt(budget_a3)));
		for (int a4 = -max_a4; a4 <= max_a4; ++a4){
			int budget_a4 = budget_a3 - 3*a4*a4;
			if (budget_a4 < 0) continue;
			
			int max_a5 = static_cast<int>(ceil(sqrt(budget_a4)));
			for (int a5 = -max_a5; a5 <= max_a5; ++a5) {
				int budget_a5 = budget_a4 - 3*a5*a5;
				if (budget_a5 < 0) continue;

				// Triangle-inequality outer pruning at (a3, a4, a5).
				// After (a3, a4, a5) are fixed, the remaining freedom lives in (a0,a1,a2),
				// bounded by parity: |a_k| <= (sqrt(budget_a5) + |a_{k+3}|)/2.
				// Compute the partial principal-embedding contribution from (a3,a4,a5),
				// bound the remaining contribution from (a0,a1,a2) via triangle inequality,
				// and derive a lower bound on |x - T|^2 for each of three target cones.
				// If no target can be reached within eps_cond, skip the (b0,b1,b2) subtree.
				{
					double sqrt_bu = sqrt((double)budget_a5);
					double max_a0 = 0.5 * (sqrt_bu + std::abs((double)a3));
					double max_a1 = 0.5 * (sqrt_bu + std::abs((double)a4));
					double max_a2 = 0.5 * (sqrt_bu + std::abs((double)a5));

					double re_partial_345 = cos_vals[3]*a3 + cos_vals[4]*a4 + cos_vals[5]*a5;
					double im_partial_345 = sin_vals[3]*a3 + sin_vals[4]*a4 + sin_vals[5]*a5;

					// cos_vals[0]=1, sin_vals[0]=0; use absolute values for triangle bound.
					double max_re_012 = max_a0
						+ std::abs(cos_vals[1]) * max_a1
						+ std::abs(cos_vals[2]) * max_a2;
					double max_im_012 = std::abs(sin_vals[1]) * max_a1
						+ std::abs(sin_vals[2]) * max_a2;

					double re_p = re_partial_345 * inv_3f;
					double im_p = im_partial_345 * inv_3f;
					double max_rem = sqrt(max_re_012*max_re_012 + max_im_012*max_im_012) * inv_3f;

					auto min_dist_sq = [&](double Tr, double Ti) -> double {
						double dx = re_p - Tr;
						double dy = im_p - Ti;
						double d_partial = sqrt(dx*dx + dy*dy);
						double d_min = d_partial - max_rem;
						if (d_min < 0.0) d_min = 0.0;
						return d_min * d_min;
					};

					double thr = eps_cond * (1.0 + FILTER_REL_TOL);
					double T1r = target_ang.real(), T1i = target_ang.imag();
					bool any_could_pass =
						   (min_dist_sq(T1r, T1i) <= thr)   // x1 cone: e^{i theta/2}
						|| (min_dist_sq(-1.0, 0.0) <= thr)  // x2 cone: -1
						|| (min_dist_sq( 0.0, 0.0) <= thr); // x3/lookup cone: 0
					if (!any_could_pass) continue;
				}

				int max_b = static_cast<int>(ceil(sqrt(budget_a5)));
				// Speedup A (2026-05-07): parity stride.  The check
				// `(b_i + a_{i+3}) % 2 != 0 → continue` rejects half the
				// iterates; instead start b_i at the lowest value ≥ -max_b
				// having the right parity and step by 2.  Saves 2× per b
				// loop = 8× total over (b0, b1, b2).
				//
				// Speedup B (2026-05-07): Schnorr-Euchner zig-zag ordering.
				// Visit b values center-out (center, center+2, center-2,
				// center+4, ...) where center is the parity-correct integer
				// closest to 0 (the b-value minimizing |b|^2, a heuristic
				// "most-promising" center).  Hits viable candidates faster
				// for paths that early-exit on first pass; same iterates
				// visited as the linear loop, just reordered.  Inline
				// up/down counters avoid per-call heap allocation.
				int b0_center = ((a3 % 2) + 2) % 2;          // 0 if a3 even, 1 if a3 odd
				if (b0_center > max_b) b0_center = b0_center - 2; // clamp into [-max_b, max_b]
				int b0_up = b0_center;        // next "up" value to emit (or > max_b if exhausted)
				int b0_dn = b0_center - 2;    // next "down" value to emit (or < -max_b if exhausted)
				bool b0_emit_up = true;       // alternation flag
				while (b0_up <= max_b || b0_dn >= -max_b) {
					int b0;
					if (b0_emit_up && b0_up <= max_b) {
						b0 = b0_up;
						b0_up += 2;
						b0_emit_up = false;
					} else if (!b0_emit_up && b0_dn >= -max_b) {
						b0 = b0_dn;
						b0_dn -= 2;
						b0_emit_up = true;
					} else if (b0_up <= max_b) {
						b0 = b0_up;
						b0_up += 2;
					} else {
						b0 = b0_dn;
						b0_dn -= 2;
					}
					int budget_b0 = budget_a5 - b0*b0;
					if (budget_b0 < 0) continue;
					int a0 = (b0 + a3) / 2;

					int max_b1 = static_cast<int>(ceil(sqrt(budget_b0)));
					int b1_center = ((a4 % 2) + 2) % 2;
					if (b1_center > max_b1) b1_center = b1_center - 2;
					int b1_up = b1_center;
					int b1_dn = b1_center - 2;
					bool b1_emit_up = true;
					while (b1_up <= max_b1 || b1_dn >= -max_b1) {
						int b1;
						if (b1_emit_up && b1_up <= max_b1) {
							b1 = b1_up;
							b1_up += 2;
							b1_emit_up = false;
						} else if (!b1_emit_up && b1_dn >= -max_b1) {
							b1 = b1_dn;
							b1_dn -= 2;
							b1_emit_up = true;
						} else if (b1_up <= max_b1) {
							b1 = b1_up;
							b1_up += 2;
						} else {
							b1 = b1_dn;
							b1_dn -= 2;
						}

						if(interrupted) goto next_a3;

						int budget_b1 = budget_b0 - b1*b1;
						if (budget_b1 < 0) continue;
						int a1 = (b1 + a4) / 2;

						int max_b2 = static_cast<int>(ceil(sqrt(budget_b1)));
						int b2_center = ((a5 % 2) + 2) % 2;
						if (b2_center > max_b2) b2_center = b2_center - 2;
						int b2_up = b2_center;
						int b2_dn = b2_center - 2;
						bool b2_emit_up = true;
						while (b2_up <= max_b2 || b2_dn >= -max_b2) {
							int b2;
							if (b2_emit_up && b2_up <= max_b2) {
								b2 = b2_up;
								b2_up += 2;
								b2_emit_up = false;
							} else if (!b2_emit_up && b2_dn >= -max_b2) {
								b2 = b2_dn;
								b2_dn -= 2;
								b2_emit_up = true;
							} else if (b2_up <= max_b2) {
								b2 = b2_up;
								b2_up += 2;
							} else {
								b2 = b2_dn;
								b2_dn -= 2;
							}
							int a2 = (b2 + a5) / 2;

							int arr[9] = {a0, a1, a2, a3, a4, a5, 0, 0, 0};
							ringZ9 local_test(arr);

							double re_raw = 0.0, im_raw = 0.0;
							for (int k = 0; k < 6; ++k) {
								re_raw += cos_vals[k] * arr[k];
								im_raw += sin_vals[k] * arr[k];
							}
							double re = re_raw * inv_3f;
							double im = im_raw * inv_3f;
							double abs_sq = re*re + im*im;

							{
								double dx = re - target_re;
								double dy = im - target_im;
								if (dx*dx + dy*dy < eps_cond * (1.0 + FILTER_REL_TOL)){
									// Mod-3 filter on x_1 candidates only (Kalra-Mosca-Valluri
									// 2023 Theorem 3.7: f(1) ≡ 0 mod 3 derivative-vanishing).
									// arr[0..5] are the ringZ9 numerator coefs for x_1.
									// Empirically retains 9.8% of x_1, preserves top-30% in
									// 96% of cells (memory hrsa_mod3_filter_pending.md).
									if (!g_hrsa_mod3_filter || ((((arr[0]+arr[1]+arr[2]+arr[3]+arr[4]+arr[5]) % 3) + 3) % 3) == 0){
										loc_x1.push_back(local_test);
									}
								}
							}
							{
								double dx = re + 1.0;
								double dy = im;
								if (dx*dx + dy*dy < eps_cond * (1.0 + FILTER_REL_TOL)){
									loc_x2.push_back(local_test);
								}
							}
							if( abs_sq < eps_cond * (1.0 + FILTER_REL_TOL)){
								loc_lookup[local_test.quad()].push_back(local_test);
							}
						} // b2
					} // b1
				} // b0
			} // a5
		} // a4
		next_a3:; // label for interrupted early-exit from b1 loop
	} // a3 (parallel)

	// Merge thread-local results into the shared output containers.
	for(int t = 0; t < omp_get_max_threads(); ++t){
		x1_cands.insert(x1_cands.end(), tl_x1[t].begin(), tl_x1[t].end());
		x2_cands.insert(x2_cands.end(), tl_x2[t].begin(), tl_x2[t].end());
		for(auto& kv : tl_lookup[t]){
			auto& dest = lookup[kv.first];
			dest.insert(dest.end(), kv.second.begin(), kv.second.end());
		}
	}
	
	// Sort x1_cands by distance to e^(i*theta/2) and x2_cands by distance to -1.
	// This causes the algorithm to try the most promising candidates first, which
	// tends to find a passing triple earlier — often at a lower f-level than the
	// unsorted original.
	//
	// Both sorts are safe with respect to correctness: they reorder candidates but
	// never remove any. For a given f, every valid (x1,x2,x3) triple is still
	// reachable. The sort cannot cause the algorithm to advance to a higher f than
	// necessary. It may return a DIFFERENT valid triple at the same f (potentially
	// with a larger eps_diff than the original), but that triple still passes epsTest.
	sort(x1_cands.begin(), x1_cands.end(), [&target_ang](const ringZ9& a, const ringZ9& b) {
		return abs(a.toComplexDouble() - target_ang) < abs(b.toComplexDouble() - target_ang);
	});
	complex<double> neg_one(-1.0, 0.0);
	sort(x2_cands.begin(), x2_cands.end(), [&neg_one](const ringZ9& a, const ringZ9& b) {
		return abs(a.toComplexDouble() - neg_one) < abs(b.toComplexDouble() - neg_one);
	});
	// lookup is a hash map; order is irrelevant for O(1) lookups by key.
	
	cout << "Finished enumerating entry possibilities for f = " << f << "." << endl;
}


bool epsTest(ringZ9chi u_1, ringZ9chi u_2, ringZ9chi u_3, int f, complex<double> angle_dir, double epsilon, double c){

	complex<double> u1_z = u_1.toComplexDouble();
	double dx = u1_z.real() - angle_dir.real();
	double dy = u1_z.imag() - angle_dir.imag();
	double eps_diff = dx*dx + dy*dy
	                + (u_2 + ringZ9chi(ringZ9(1),0)).abs_val_sq()
	                + u_3.abs_val_sq();
	double eps_cond = epsilon*epsilon/(8.0*c*c);

	// No cout here: this function is called from parallel threads and must be
	// lock-free. The caller logs the result once a solution is confirmed.
	return eps_diff < eps_cond;
}


int three_power(int n){
	
	int prod = 1;
	
	if(n < 0){
		cout << "Negative argument passed to three_power(int n)" << endl;
		return 0;
	}
	
	for(int i = 0; i < n; i++){
		prod = prod*3;
	}
	
	return prod;
	
}

void handleCtrlC(int sig) {
    interrupted = true;
}

void matrixFrobeniusCheck(const std::array<ringZ9chi,3>& u, double theta, double epsilon){
	// Build H = I_3 - u * u^dagger as a 3x3 complex<double> matrix.
	// u^dagger means conjugate-transpose; since u is a column vector,
	// H_{ij} = delta_{ij} - u_i * conj(u_j).
	using cd = complex<double>;
	array<cd,3> uc = {
		u[0].toComplexDouble(),
		u[1].toComplexDouble(),
		u[2].toComplexDouble()
	};

	// H[i][j] = delta_ij - uc[i]*conj(uc[j])
	cd H[3][3];
	for(int i = 0; i < 3; ++i)
		for(int j = 0; j < 3; ++j)
			H[i][j] = (i==j ? cd(1,0) : cd(0,0)) - uc[i] * conj(uc[j]);

	// X_{(0,1)} = [[0,1,0],[1,0,0],[0,0,1]]  (SWAP of rows 0 and 1).
	// This is the fixed correction unitary such that X_{(0,1)} * H ~ R_{(0,1)}^Z(theta)
	// when u ~ [e^{itheta/2}, -1, 0]. It is theta-independent.
	// XH[i][j] = sum_k X[i][k] * H[k][j]; with X = SWAP this exchanges rows 0 and 1:
	cd XH[3][3];
	for(int j = 0; j < 3; ++j){
		XH[0][j] = H[1][j];  // row 0 of X*H = row 1 of H
		XH[1][j] = H[0][j];  // row 1 of X*H = row 0 of H
		XH[2][j] = H[2][j];  // row 2 unchanged
	}

	// R_{(0,1)}^Z(theta) = Diag(e^{-itheta/2}, e^{itheta/2}, 1).
	// Canonical convention (theta as user passes it).
	cd R[3][3] = {};
	R[0][0] = polar(1.0, -theta/2.0);
	R[1][1] = polar(1.0,  theta/2.0);
	R[2][2] = cd(1.0, 0.0);

	// Frobenius norm: ||XH - R||_F = sqrt(sum_{ij} |XH_{ij} - R_{ij}|^2)
	double frob_sq = 0.0;
	for(int i = 0; i < 3; ++i)
		for(int j = 0; j < 3; ++j){
			cd diff = XH[i][j] - R[i][j];
			frob_sq += diff.real()*diff.real() + diff.imag()*diff.imag();
		}
	double frob = sqrt(frob_sq);

	cout << "Matrix Frobenius distance ||X_(0,1) H - R_(0,1)^Z(theta)||_F = "
	     << frob << "   (target epsilon = " << epsilon << ", passes: "
	     << (frob < epsilon ? "YES" : "NO") << ")" << endl;
}

array<ringZ9chi,3> HRSA_bestD(double theta, double epsilon, int max_f, double c, int max_solns, int k3){
	int f = 0;
	ringZ9chi zero;
	array<ringZ9chi,3> answer;
	vector<ringZ9> x1_cands, x2_cands;
	unordered_map<int,vector<ringZ9>> lookup;

	// Convention: theta is canonical (target = R^Z_{(0,1)}(theta));
	// search target for x_1 is e^{+i theta/2}.
	complex<double> angle_dir(cos(theta/2.0), sin(theta/2.0));

	answer[0] = zero; answer[1] = zero; answer[2] = zero;

	cout << "HRSA_bestD: max_solns=" << max_solns
	     << ", OMP threads=" << omp_get_max_threads() << endl;

	while( f <= max_f){
		int f_pow_sq = three_power(f) * three_power(f);
		ringZ9 f_pow_sq_doub(2*f_pow_sq);

		cout << "Enumerating entry candidates for f = " << f << "." << endl;
		entryEnumeration(x1_cands, x2_cands, lookup, theta, epsilon, f, c);

		// Per-conjugate norm caches for sigma_2, sigma_4 (see HRSA() above for derivation).
		const int N1 = (int)x1_cands.size();
		const int N2 = (int)x2_cands.size();
		std::vector<double> x1_s2sq(N1), x1_s4sq(N1);
		std::vector<double> x2_s2sq(N2), x2_s4sq(N2);
		#pragma omp parallel for schedule(static) if(N1 + N2 >= 1024)
		for(int idx = 0; idx < N1 + N2; ++idx){
			if(idx < N1){
				x1_s2sq[idx] = x1_cands[idx].GaloisAut(2).abs_val_sq();
				x1_s4sq[idx] = x1_cands[idx].GaloisAut(4).abs_val_sq();
			} else {
				int j = idx - N1;
				x2_s2sq[j] = x2_cands[j].GaloisAut(2).abs_val_sq();
				x2_s4sq[j] = x2_cands[j].GaloisAut(4).abs_val_sq();
			}
		}
		const double conj_thr = 2.0 * (double)f_pow_sq * (1.0 + FILTER_REL_TOL);

		// Collect up to max_solns valid candidates at this f level.
		vector<array<ringZ9chi,3>> solutions;
		atomic<int> soln_count(0);
		mutex soln_mutex;

		// Iteration order over the outer x_1 loop.  Three modes:
		//   1) g_hrsa_rf_gate > 0: union(distance-top-K, RF-top-K) — Phase 3.5.
		//      For each x_1, score with the trained RandomForest using the matching
		//      x_2 candidates as the sub-cluster.  Iterate the union of the
		//      distance-top-K and RF-top-K candidates.  Skip duplicates.
		//   2) g_hrsa_alt_order: front-back alternation (default OFF; A/B was -ve).
		//   3) sequential (default).
		std::vector<int> i_order;
		i_order.reserve(N1);
		if(g_hrsa_rf_gate > 0 && N1 > 2 * g_hrsa_rf_gate){
			// Compute RF score for each x_1.  For each x_1, gather matching x_2's
			// (those passing the per-conjugate filter — same condition used in the
			// inner loop below).
			std::vector<double> rf_scores(N1, 0.0);
			std::vector<int> n_in_cluster(N1, 0);
			#pragma omp parallel for schedule(dynamic) if(N1 >= 32)
			for(int i = 0; i < N1; ++i){
				const ringZ9& x_1 = x1_cands[i];
				int q1 = x_1.quad();
				const double s2_x1 = x1_s2sq[i];
				const double s4_x1 = x1_s4sq[i];
				std::vector<ringZ9> matched;
				matched.reserve(8);
				for(int j = 0; j < N2; ++j){
					int q2 = x2_cands[j].quad();
					if(q1 + q2 > 2 * f_pow_sq) continue;
					if(s2_x1 + x2_s2sq[j] > conj_thr) continue;
					if(s4_x1 + x2_s4sq[j] > conj_thr) continue;
					matched.push_back(x2_cands[j]);
				}
				n_in_cluster[i] = (int)matched.size();
				if(matched.empty()){
					rf_scores[i] = 1e9;  // bogus high so it gets deprioritised
				} else {
					rf_scores[i] = rf_score_subcluster(x_1, matched, f, (int)matched.size());
				}
			}
			// Distance-top-K = first K indices (already sorted by distance to target).
			// RF-top-K = K indices with smallest rf_scores.
			int K = g_hrsa_rf_gate;
			if(K > N1) K = N1;
			std::vector<int> rf_order(N1);
			for(int i = 0; i < N1; ++i) rf_order[i] = i;
			std::sort(rf_order.begin(), rf_order.end(),
				[&](int a, int b){ return rf_scores[a] < rf_scores[b]; });
			std::vector<bool> seen(N1, false);
			// Interleave: dist[0], rf[0], dist[1], rf[1], ...
			for(int k = 0; k < K; ++k){
				if(!seen[k]){ i_order.push_back(k); seen[k] = true; }
				int rfi = rf_order[k];
				if(!seen[rfi]){ i_order.push_back(rfi); seen[rfi] = true; }
			}
			cout << "[Phase 3.5] RF-gate K=" << K << " gives "
			     << i_order.size() << " unique x_1's "
			     << "(union of distance-top-" << K << " and RF-top-" << K << ")" << endl;
		} else if(g_hrsa_alt_order){
			int lo = 0, hi = N1 - 1;
			while(lo <= hi){
				i_order.push_back(lo++);
				if(lo <= hi) i_order.push_back(hi--);
			}
		} else {
			for(int i = 0; i < N1; ++i) i_order.push_back(i);
		}

		#pragma omp parallel for schedule(dynamic) if(f >= 2) \
		    shared(soln_count, soln_mutex, solutions, i_order)
		for(int idx_i = 0; idx_i < (int)i_order.size(); ++idx_i){

			if(soln_count >= max_solns || interrupted) continue;

			int i = i_order[idx_i];
			const ringZ9& x_1 = x1_cands[i];
			int q1 = x_1.quad();
			const double s2_x1 = x1_s2sq[i];
			const double s4_x1 = x1_s4sq[i];

			for(int j = 0; j < (int)x2_cands.size(); ++j){

				if(soln_count >= max_solns || interrupted) break;

				const ringZ9& x_2 = x2_cands[j];
				int q2 = x_2.quad();
				if(q1 + q2 > 2 * f_pow_sq) continue;
				if(s2_x1 + x2_s2sq[j] > conj_thr) continue;
				if(s4_x1 + x2_s4sq[j] > conj_thr) continue;

				int target_norm = 2*f_pow_sq - q1 - q2;
				ringZ9 x3sq = f_pow_sq_doub - x_1.complexConj()*x_1 - x_2.complexConj()*x_2;

				auto bucket_it = lookup.find(target_norm);
				if(bucket_it == lookup.end()) continue;
				// K_3 cap: collect at most k3 valid x_3 hits per (x_1, x_2) pair.
				// Within a (x_1, x_2) sub-cluster, the x_3 candidates form a 9-fold
				// zeta_9 orbit of essentially the same triple, with N_D nearly constant
				// (~5-gate spread).  Across sub-clusters N_D varies by 16-39 gates.
				// Capping x_3 hits per pair forces the budget to spend on diverse
				// (x_1, x_2) sub-clusters instead of redundant x_3 variants.
				int hits_this_pair = 0;
				for(const ringZ9& x_3 : bucket_it->second){

					if(soln_count >= max_solns) break;
					if(hits_this_pair >= k3) break;

					if(!(x_3.complexConj()*x_3 == x3sq)) continue;

					array<ringZ9chi,3> candidate = {
						ringZ9chi(x_1,f),
						ringZ9chi(x_2,f),
						ringZ9chi(x_3,f)
					};

					if(epsTest(candidate[0], candidate[1], candidate[2], f, angle_dir, epsilon, c)){
						lock_guard<mutex> lk(soln_mutex);
						if(soln_count < max_solns){
							solutions.push_back(candidate);
							soln_count++;
							hits_this_pair++;
							// 2026-05-13: emit each hit to stderr so external watchers
							// can see search progress mid-run (useful for tuning
							// max-solns / timeout on tight-ε cells).
							std::cerr << "[hit] f=" << f
							          << " soln_count=" << soln_count
							          << "/" << max_solns << std::endl;
						}
					}
				}
			}
		}

		if(!solutions.empty()){
			cout << "Found " << solutions.size() << " candidate(s) at f=" << f
			     << ". Decomposing each to find minimum D-count..." << endl;

			int n_solns = (int)solutions.size();
			vector<int> d_counts(n_solns, INT_MAX);

			// Each decompose() call is independent — parallelize across candidates.
			// cout inside decompose() may interleave, but correctness is unaffected.
			#pragma omp parallel for schedule(dynamic)
			for(int s = 0; s < n_solns; ++s){
				Mat3 V = buildUnitary(solutions[s]);
				DecompResult dr = decompose(V, true);  // quiet: suppress cout in parallel
				d_counts[s] = dr.success ? dr.D_count : INT_MAX;
			}

			// Find best and report (serial, so output is clean)
			int best_idx = 0;
			int best_D = INT_MAX;
			for(int s = 0; s < n_solns; ++s){
				cout << "  Candidate " << (s+1) << "/" << n_solns
				     << ": D_gates=" << (d_counts[s] < INT_MAX ? to_string(d_counts[s]) : "FAIL") << endl;
				// CANDDUMP: per-candidate machine-readable dump for diversity analysis.
				// Format: CANDDUMP s D_gates x1_re x1_im x2_re x2_im x3_re x3_im
				//         f a1_0..a1_5 a2_0..a2_5 a3_0..a3_5
				// (18 integer numerator coefficients appended for ML / algebraic features.)
				auto cdx1 = solutions[s][0].toComplexDouble();
				auto cdx2 = solutions[s][1].toComplexDouble();
				auto cdx3 = solutions[s][2].toComplexDouble();
				ringZ9 num1 = solutions[s][0].getNumerator();
				ringZ9 num2 = solutions[s][1].getNumerator();
				ringZ9 num3 = solutions[s][2].getNumerator();
				cout << "CANDDUMP " << s << " "
				     << (d_counts[s] < INT_MAX ? d_counts[s] : -1) << " "
				     << cdx1.real() << " " << cdx1.imag() << " "
				     << cdx2.real() << " " << cdx2.imag() << " "
				     << cdx3.real() << " " << cdx3.imag() << " "
				     << f;
				for(int kk = 0; kk < 6; ++kk) cout << " " << num1.getTerm(kk);
				for(int kk = 0; kk < 6; ++kk) cout << " " << num2.getTerm(kk);
				for(int kk = 0; kk < 6; ++kk) cout << " " << num3.getTerm(kk);
				cout << endl;
				if(d_counts[s] < best_D){
					best_D = d_counts[s];
					best_idx = s;
				}
			}

			answer = solutions[best_idx];

			double eps_diff = pow(abs(answer[0].toComplexDouble() - angle_dir),2)
			                + (answer[1] + ringZ9chi(ringZ9(1),0)).abs_val_sq()
			                + answer[2].abs_val_sq();
			cout << "Epsilon Diff. Val.: " << eps_diff
			     << " Eps. Cond.: " << epsilon*epsilon/(8.0*c*c) << endl;
			cout << "Selected candidate " << (best_idx+1) << " with "
			     << best_D << " D-gate(s)." << endl;
			cout << "Success!" << endl;
			return answer;
		}
		if(interrupted) return answer;

		f++;
		x1_cands.clear();
		x2_cands.clear();
		lookup.clear();
	}
	cout << "Failure." << endl;
	return answer;
}
