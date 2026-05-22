#include<iostream>
#include<fstream>
#include "householder_search.h"
#include "decompose.h"
#include "direct_skip_table.h"
#include "bidir_reify.h"
#include "clifford_cache.h"
#include<vector>
#include<array>
#include<cmath>
#include<complex>
#include<numeric>
#include<atomic>
#include<csignal>
#include<cstring>
#include<chrono>
#include<unistd.h>
#include<ctime>
#include<sstream>
#include<omp.h>
#include "cyclotomic_int9.h"
#include "Z9chi.h"


std::atomic<bool> interrupted(false); // For clean exiting with Ctrl+C if execution takes too long

// Toggled by `--no-simplify`.  Defined in decompose.cpp.
extern bool g_decompose_simplify;
extern bool g_decompose_lookahead;
extern bool g_hrsa_alt_order;
extern int  g_hrsa_rf_gate;
extern bool g_hrsa_mod3_filter;

using namespace std;

int main(int argc, char* argv[]){
	
	signal(SIGINT, handleCtrlC);
	
	if(argc < 4){
		cout << "Usage: ./HRSA_tester theta epsilon max_f [c] [flags]" << endl;
		cout << "  theta                : target rotation angle" << endl;
		cout << "  epsilon              : desired precision" << endl;
		cout << "  max_f                : maximum f level to search" << endl;
		cout << "  c                    : contraction factor in (0,1], default 1.0" << endl;
		cout << "  --max-solns N        : collect N HRSA candidates, pick lowest D-count" << endl;
		cout << "  --max-direct K       : direct search up to K D-gates (default 2, 0=Clifford only)" << endl;
		cout << "  --no-direct          : skip direct search, go straight to HRSA" << endl;
		cout << "  --use-bidir MAX_K    : enable Phase 2 bidirectional BFS (MAX_K ∈ {3,4,5}; off by default)" << endl;
		cout << "  --no-simplify        : disable post-pass rewrite simplifier on decompose()" << endl;
		cout << "  --json PATH          : write JSON result to PATH" << endl;
		cout << "  --fast               : bundle for the bidir↔zeta9 middle-ground heuristic:" << endl;
		cout << "                         max-solns=1 + no-lookahead. Faster, slightly worse N_D." << endl;
		cout << "  --dump-x1-cands PATH : enumerate-only fast path; dump u_1 candidates as" << endl;
		cout << "                         JSON lines to PATH and exit (skips both phases)" << endl;
		cout << "  --mod3-filter        : Kalra-Mosca-Valluri 2023 Thm 3.7 prune on x_1 candidates" << endl;
		cout << "                         (sum(a_1) ≡ 0 mod 3); ~10× fewer candidates, est ~20-40× speedup" << endl;
		return 1;
	}

	double theta = atof(argv[1]);
	double epsilon = atof(argv[2]);
	int max_f = atoi(argv[3]);
	double c = 1.0;
	int max_solns = 1;  // default: original behavior
	int max_direct = 2; // default: direct search up to k=2
	int use_bidir = 0;  // --use-bidir MAX_K; 0 = disabled (default)
	int k3 = 1;         // x_3 hits per (x_1, x_2) pair in HRSA_bestD; 1 = stratified
	string json_path = "";  // --json output path
	string dump_x1_path = ""; // --dump-x1-cands output path (enumerate-only fast path)

	// Parse optional positional arg (c) and flags
	for(int i = 4; i < argc; ++i){
		if(strcmp(argv[i], "--max-solns") == 0 && i+1 < argc){
			max_solns = atoi(argv[++i]);
			if(max_solns < 1) max_solns = 1;
		} else if(strcmp(argv[i], "--max-direct") == 0 && i+1 < argc){
			max_direct = atoi(argv[++i]);
			if(max_direct < 0) max_direct = 0;
		} else if(strcmp(argv[i], "--no-direct") == 0){
			max_direct = -1;  // skip direct search entirely
		} else if(strcmp(argv[i], "--use-bidir") == 0 && i+1 < argc){
			use_bidir = atoi(argv[++i]);
			if(use_bidir < 0) use_bidir = 0;
		} else if(strcmp(argv[i], "--k3") == 0 && i+1 < argc){
			k3 = atoi(argv[++i]);
			if(k3 < 1) k3 = 1;
		} else if(strcmp(argv[i], "--no-simplify") == 0){
			g_decompose_simplify = false;
		} else if(strcmp(argv[i], "--no-lookahead") == 0){
			g_decompose_lookahead = false;
		} else if(strcmp(argv[i], "--alt-order") == 0){
			g_hrsa_alt_order = true;
		} else if(strcmp(argv[i], "--mod3-filter") == 0){
			// Kalra-Mosca-Valluri 2023 Theorem 3.7 derivative-vanishing
			// condition on x_1 candidates. Empirically prunes ~90% of x_1
			// with strong N_D bias. See memory hrsa_mod3_filter_pending.md.
			g_hrsa_mod3_filter = true;
		} else if(strcmp(argv[i], "--rf-gate") == 0 && i+1 < argc){
			g_hrsa_rf_gate = atoi(argv[++i]);
			if(g_hrsa_rf_gate < 0) g_hrsa_rf_gate = 0;
		} else if(strcmp(argv[i], "--json") == 0 && i+1 < argc){
			json_path = argv[++i];
		} else if(strcmp(argv[i], "--fast") == 0){
			// Bundle: faster-but-suboptimal heuristic between bidir and full HRSA.
			// Future additions to this bundle should be performance-only (not
			// changing semantics or hidden flags that affect correctness).
			max_solns = 1;
			g_decompose_lookahead = false;
		} else if(strcmp(argv[i], "--dump-x1-cands") == 0 && i+1 < argc){
			dump_x1_path = argv[++i];
		} else if(argv[i][0] != '-'){
			// positional: must be c
			c = atof(argv[i]);
			if (c <= 0.0 || c > 1.0){
				cout << "Invalid choice of c. c must lie in interval (0,1]." << endl;
				return 1;
			}
		}
	}

	// =========================================================
	//  Fast path: --dump-x1-cands enumerate-only mode
	//  Runs ONLY entryEnumeration() for each f in [0, max_f] and
	//  writes the resulting u_1 candidates as JSON lines, then exits.
	//  This lets a downstream wrapper (e.g. zeta9) consume the u_1
	//  list and complete u_2 itself without paying for the HRSA
	//  triple-product matching phase.
	// =========================================================
	if (!dump_x1_path.empty()) {
		ofstream js(dump_x1_path);
		if (!js){
			cerr << "ERROR: cannot open --dump-x1-cands path: " << dump_x1_path << endl;
			return 2;
		}

		long long total = 0;
		for (int f = 0; f <= max_f; ++f) {
			vector<ringZ9> x1_cands, x2_cands;
			unordered_map<int,vector<ringZ9>> lookup;
			cout << "Enumerating entry candidates for f = " << f << "." << endl;
			entryEnumeration(x1_cands, x2_cands, lookup, theta, epsilon, f, c);
			cout << "  f=" << f << ": |x1_cands|=" << x1_cands.size() << endl;
			for (const ringZ9& x : x1_cands) {
				array<int,6> a = x.getStdArray();
				js << "{\"f\": " << f << ", \"u\": ["
				   << a[0] << ", " << a[1] << ", " << a[2] << ", "
				   << a[3] << ", " << a[4] << ", " << a[5] << "]}\n";
				++total;
			}
			if (interrupted) break;
		}
		js.close();
		cout << "Wrote " << total << " u_1 candidates to " << dump_x1_path << endl;
		return 0;
	}

	// Wall-clock start for performance reporting.
	auto wall_start = chrono::steady_clock::now();

	// Track results across all phases for the summary block / JSON output.
	string method = "none";
	int result_D = -1;
	double result_frob = -1.0;
	int result_f = -1;        // f from output u-vector (for Phase 3) or 0 (Phase 1)
	bool found = false;
	Mat3 result_V;            // populated on success (either phase)
	bool result_V_set = false;
	int result_v_f = -1;      // Common denominator exp for the V matrix entries (for JSON)
	DecompResult result_dr;   // decompose result (Phase 3); steps available
	bool result_dr_set = false;

	// --- Phase 0.5: Sign-extended Clifford lookup ---
	// For each of 8 sign matrices S ∈ {±I, ±R, ±R', ±R''} and each of the 648
	// Cliffords C, compute Frobenius distance ‖S·C − R^Z(theta)‖_F.  If the
	// minimum is below epsilon, we have a sign-extended-Clifford solution
	// with N_D = (S != I ? 1 : 0).  This catches cheap solutions that bidir
	// (which doesn't include R) misses — e.g., diag(-ω, -ω², 1) at θ=2.14.
	if (!found) {
		using cd = complex<double>;
		const auto& cache = get_clifford_cache();
		const auto& cliffs_c = cache.cliffords;  // CMat3 (floating-point) cache
		cd target[3][3] = {};
		target[0][0] = polar(1.0, -theta/2.0);
		target[1][1] = polar(1.0,  theta/2.0);
		target[2][2] = cd(1.0, 0.0);

		double best_frob = 1e30;
		int best_clifford_idx = -1;
		int best_sign_pattern = -1;  // 3 bits: bit i = sign of slot i (1 = -)
		for (int sp = 0; sp < 8; ++sp) {
			double s0 = (sp & 1) ? -1.0 : 1.0;
			double s1 = (sp & 2) ? -1.0 : 1.0;
			double s2 = (sp & 4) ? -1.0 : 1.0;
			for (size_t k = 0; k < cliffs_c.size(); ++k) {
				const auto& C = cliffs_c[k];
				double frob_sq = 0;
				for (int i = 0; i < 3; ++i) {
					double si = (i==0?s0:(i==1?s1:s2));
					for (int j = 0; j < 3; ++j) {
						cd v = si * C.m[i][j] - target[i][j];
						frob_sq += norm(v);
					}
				}
				double frob = sqrt(frob_sq);
				if (frob < best_frob) { best_frob = frob; best_clifford_idx = (int)k; best_sign_pattern = sp; }
			}
		}
		cout << "\n=== Phase 0.5: Sign-extended Clifford lookup ===" << endl;
		cout << "Best Frob = " << best_frob << "  (eps = " << epsilon << ")"
		     << "  cliff_idx=" << best_clifford_idx << "  sign_pattern=" << best_sign_pattern << endl;
		if (best_frob < epsilon) {
			method = "SignExtClifford";
			result_D = (best_sign_pattern == 0 ? 0 : 1);
			result_frob = best_frob;
			result_f = 0;
			// Reify V exactly over ringZ9chi: V = sign_matrix · ring_clifford.
			Mat3 sign_M;
			ringZ9 one_z(1), neg_one_z(-1);
			ringZ9chi p1(one_z, 0), m1(neg_one_z, 0);
			sign_M.m[0][0] = (best_sign_pattern & 1) ? m1 : p1;
			sign_M.m[1][1] = (best_sign_pattern & 2) ? m1 : p1;
			sign_M.m[2][2] = (best_sign_pattern & 4) ? m1 : p1;
			result_V = sign_M.mul(cache.ring_cliffords[best_clifford_idx]);
			result_V_set = true;
			found = true;
			cout << "Phase 0.5 success!  N_D = " << result_D
			     << "  frob = " << result_frob << endl;
		}
	}

	// --- Phase 1: Direct gate-sequence search for low D-count ---
	if (max_direct >= 0 && !found) {
		// Skip if calibrated cum_max_d(theta) > epsilon: no Clifford+kD circuit exists
		// within tolerance.  Clifford-Z angles always run direct (exact match at k=0).
		if (direct_skip::should_skip_direct(epsilon, theta, max_direct)) {
			cout << "\n=== Phase 1: Direct search SKIPPED (eps=" << epsilon
			     << " < cum_" << max_direct << "(theta)~" << direct_skip::lookup_cum(theta, max_direct)
			     << ", non-Clifford angle) ===" << endl;
		} else {
		cout << "\n=== Phase 1: Direct search (D-count 0-" << max_direct << ") ===" << endl;
		DirectSearchResult dsr = directSearch(theta, epsilon, max_direct);
	
		if (dsr.success) {
			method = "Direct";
			result_D = dsr.D_count;
			result_frob = dsr.frob_dist;
			result_f = 0;
			result_V = dsr.V;
			result_V_set = true;
			found = true;
		} else {
			cout << "\nNo low-D solution found via direct search." << endl;
		}
		}  // end else (skip predicate did not fire)
	}

	// --- (Original Phase 2 / diagSearch removed 2026-05) ---
	// The original Phase 2 (diagSearch) was a heuristic that:
	//   (a) found NON-UNITARY V's by lattice search with no unitarity check;
	//   (b) ran decompose() which took a diagonal-V fast-path returning a
	//       heuristic D-count with NO syllables produced;
	//   (c) reported V's Frobenius distance as if V were the implementable
	//       circuit — but no actual Clifford+D circuit exists for the
	//       returned V.
	// Removed entirely; the bidirectional BFS approximation below now
	// occupies the Phase 2 slot and produces exactly-unitary V's.

	// --- Phase 2: Bidirectional BFS approximation (opt-in via --use-bidir) ---
	if (use_bidir > 0 && !found) {
		// Threshold (calibrated 2026-05-09 over 8 thetas, full sweep):
		//   K=4 (depth  8) worst-case Frob = 0.2530 at θ=2.14
		//   K=5 (depth 10) worst-case Frob = 0.1853 at θ=2.14
		//   K=6 (depth 12) worst-case Frob ≈ 0.096 (too slow for production)
		// Pick the smallest K such that worst-case Frob < epsilon.
		int K_needed = -1;
		if (epsilon > 0.2530)      K_needed = 4;
		else if (epsilon > 0.1853) K_needed = 5;
		// (K=6 would handle eps > 0.10 but is too slow for production.)

		if (K_needed > 0 && K_needed <= use_bidir) {
			cout << "\n=== Phase 2: Bidirectional BFS (K_f=K_b="
			     << K_needed << ") ===" << endl;
			BidirCircuit bc = run_bidir(theta, K_needed, K_needed, epsilon);
			if (bc.valid && bc.frob_to_target < epsilon) {
				method = "bidir(" + to_string(K_needed) + "+" + to_string(K_needed) + ")";
				result_D = bc.N_D;
				result_frob = bc.frob_to_target;
				result_f = bc.v_f;
				result_v_f = bc.v_f;
				result_V = bc.V;
				result_V_set = true;
				found = true;
				cout << "Phase 2 success! N_D=" << result_D
				     << " Frob=" << result_frob
				     << " v_f=" << result_v_f << endl;
			} else {
				cout << "Phase 2 bidir did not find a circuit within eps; "
				        "falling through." << endl;
			}
		} else {
			cout << "\n=== Phase 2: bidir SKIPPED (eps too tight for K<="
			     << use_bidir << ") ===" << endl;
		}
	}

	// --- Phase 3: HRSA Householder search ---
	if (!found) {
		cout << "\n=== Phase 3: HRSA search ===" << endl;
		array<ringZ9chi,3> output2;
		auto _ts_hrsa_start = chrono::steady_clock::now();
		if(max_solns > 1){
			output2 = HRSA_bestD(theta, epsilon, max_f, c, max_solns, k3);
		} else {
			output2 = HRSA(theta, epsilon, max_f, c);
		}
		auto _ts_hrsa_end = chrono::steady_clock::now();
		cerr << "[PROF] HRSA(_bestD)() wall = "
		     << chrono::duration<double>(_ts_hrsa_end - _ts_hrsa_start).count()
		     << " s" << endl;

		// Check if HRSA found something (non-zero vector)
		bool hrsa_found = !(output2[0].isZero() && output2[1].isZero() && output2[2].isZero());

		if (hrsa_found) {
			cout << "x_1: ";
			output2[0].print(); 
			cout << " " << output2[0].abs_val() << "\n";
			cout << "x_2: ";
			output2[1].print(); 
			cout << " " << output2[1].abs_val() << "\n";
			cout << "x_3: ";
			output2[2].print(); 
			cout << " " << output2[2].abs_val() << "\n";
			cout << "Target x_1 = e^(i theta/2) : " << cos(theta/2.0) << " + i" << sin(theta/2) << endl; 
			cout << "Found x_1 (should be close to Target x_1): " << real(output2[0].toComplexDouble()) << " + i" << imag(output2[0].toComplexDouble()) << endl;
			cout << "Found x_2 (should be close to -1): " << real(output2[1].toComplexDouble()) << " + i" << imag(output2[1].toComplexDouble()) << endl;
			cout << "Found x_3 (should be close to 0): " << real(output2[2].toComplexDouble()) << " + i" << imag(output2[2].toComplexDouble()) << endl;
			cout << "Squared Frobenius norm of vector (should be exactly 2): " 
			     << output2[0].abs_val_sq() + output2[1].abs_val_sq() + output2[2].abs_val_sq() << endl;

			matrixFrobeniusCheck(output2, theta, epsilon);

			auto _ts_bu = chrono::steady_clock::now();
			Mat3 V = buildUnitary(output2);
			auto _ts_dec_start = chrono::steady_clock::now();
			DecompResult dr = decompose(V, true);
			auto _ts_dec_end = chrono::steady_clock::now();
			cerr << "[PROF] buildUnitary() wall = "
			     << chrono::duration<double>(_ts_dec_start - _ts_bu).count() << " s" << endl;
			cerr << "[PROF] decompose() wall = "
			     << chrono::duration<double>(_ts_dec_end - _ts_dec_start).count() << " s" << endl;
			result_V = V;
			result_V_set = true;
			result_dr = dr;
			result_dr_set = true;

			// Compute Frobenius distance for summary

			// Get f level from the exponent
			result_f = output2[0].getExp();

			method = "HRSA(f=" + to_string(result_f) + ")";
			result_D = dr.success ? dr.D_count : -1;
			// Use Matrix Frobenius distance (computed by matrixFrobeniusCheck)
			// Recompute it here for the summary
			{
				using cd = complex<double>;
				array<cd,3> uc = {output2[0].toComplexDouble(), output2[1].toComplexDouble(), output2[2].toComplexDouble()};
				cd H[3][3];
				for(int i=0;i<3;++i) for(int j=0;j<3;++j)
					H[i][j] = (i==j ? cd(1,0) : cd(0,0)) - uc[i]*conj(uc[j]);
				cd XH[3][3];
				for(int j=0;j<3;++j){ XH[0][j]=H[1][j]; XH[1][j]=H[0][j]; XH[2][j]=H[2][j]; }
				cd R[3][3]={};
				R[0][0]=polar(1.0,-theta/2.0); R[1][1]=polar(1.0,theta/2.0); R[2][2]=cd(1,0);
				double frob_sq=0;
				for(int i=0;i<3;++i) for(int j=0;j<3;++j){
					cd d=XH[i][j]-R[i][j]; frob_sq+=d.real()*d.real()+d.imag()*d.imag();
				}
				result_frob = sqrt(frob_sq);
			}
			// Only declare overall success if decompose() also succeeded.
			// HRSA can produce a unitary V that's a non-Clifford monomial
			// (e.g. diag(-ω,-ω²,1) — close to a target rotation but not in
			// the Clifford group); decompose() flags these via success=false
			// because the fast-path can't count their D-gates.  Surfacing this
			// as overall failure prevents the misleading N_D=0 / empty syllables.
			found = dr.success;
		}
	}

	// =========================================================
	//  PARSEABLE SUMMARY — parse_results.py reads these lines
	// =========================================================
	cout << "\n=== Summary ===" << endl;
	if (found) {
		cout << "Epsilon Diff. Val.: " << result_frob << " Eps. Cond.: " << epsilon*epsilon/(8.0*c*c) << endl;
		cout << "Matrix Frobenius distance ||X_(0,1) H - R_(0,1)^Z(theta)||_F = "
		     << result_frob << "   (target epsilon = " << epsilon << ", passes: "
		     << (result_frob < epsilon ? "YES" : "NO") << ")" << endl;
		cout << "Total D gates:    " << result_D << endl;
		cout << "Method: " << method << endl;
		cout << "Success!" << endl;
	} else {
		cout << "Failure." << endl;
	}

	double wall_seconds = chrono::duration<double>(
		chrono::steady_clock::now() - wall_start).count();

	// =========================================================
	//  JSON output (compile_qutrit_schema.md, v1.0)
	// =========================================================
	if (!json_path.empty()) {
		ofstream js(json_path);
		if (!js){
			cerr << "ERROR: cannot open --json path: " << json_path << endl;
			return 2;
		}

		// --- Compute V's common denominator and pad entry numerators ---
		// Each ringZ9chi entry has its own getExp(); the schema requires ONE
		// common f.  Take f_common = max over entries; pad lower-exp entries
		// by multiplying their numerator by 3^(f_common - getExp()).
		int f_common = 0;
		if (result_V_set){
			for(int i = 0; i < 3; ++i)
				for(int j = 0; j < 3; ++j)
					if (result_V.m[i][j].getExp() > f_common)
						f_common = result_V.m[i][j].getExp();
			result_v_f = f_common;
		}

		auto emit_v_entry = [&](int i, int j) -> string {
			int e = result_V.m[i][j].getExp();
			int scale = 1;
			for (int k = 0; k < f_common - e; ++k) scale *= 3;
			array<int,6> a = result_V.m[i][j].getStdArray();
			ostringstream o;
			o << "[";
			for (int k = 0; k < 6; ++k){
				if (k) o << ", ";
				o << (a[k] * scale);
			}
			o << "]";
			return o.str();
		};

		// Compute unitarity residual ‖V V† − I‖_F (floating-point) for sanity_checks.
		double unitarity_residual = 0.0;
		if (result_V_set){
			Mat3 VV = result_V.mul(result_V.dagger());
			using cd = complex<double>;
			for(int i = 0; i < 3; ++i){
				for(int j = 0; j < 3; ++j){
					cd v = VV.m[i][j].toComplexDouble();
					cd target_ij = (i == j) ? cd(1, 0) : cd(0, 0);
					cd d = v - target_ij;
					unitarity_residual += d.real()*d.real() + d.imag()*d.imag();
				}
			}
			unitarity_residual = sqrt(unitarity_residual);
		}

		// Get hostname
		char hostname[256] = {0};
		gethostname(hostname, sizeof(hostname) - 1);

		// ISO-8601 timestamp
		char tsbuf[32];
		{
			time_t now = time(nullptr);
			struct tm gmt;
			gmtime_r(&now, &gmt);
			strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%dT%H:%M:%SZ", &gmt);
		}

		// --- Emit JSON ---
		js << "{\n";
		// identification
		js << "  \"identification\": {\n";
		js << "    \"backend\": \"hrsa\",\n";
		js << "    \"version\": \"hrsa-2026-05\",\n";
		js << "    \"command_line\": [";
		for (int i = 0; i < argc; ++i){
			if (i) js << ", ";
			js << "\"" << argv[i] << "\"";
		}
		js << "],\n";
		js << "    \"host\": \"" << hostname << "\",\n";
		js << "    \"timestamp\": \"" << tsbuf << "\",\n";
		js << "    \"schema_version\": \"1.0\"\n";
		js << "  },\n";
		// inputs
		js << "  \"inputs\": {\n";
		js << "    \"theta\": " << theta << ",\n";
		js << "    \"epsilon\": " << epsilon << ",\n";
		js << "    \"max_f\": " << max_f << ",\n";
		js << "    \"c\": " << c << ",\n";
		js << "    \"max_solns\": " << max_solns << ",\n";
		js << "    \"max_direct\": " << max_direct << "\n";
		js << "  },\n";
		// achieved
		js << "  \"achieved\": {\n";
		js << "    \"success\": " << (found ? "true" : "false") << ",\n";
		if (found) {
			js << "    \"achieved_frob\": " << result_frob << ",\n";
			js << "    \"epsilon_passed\": " << (result_frob < epsilon ? "true" : "false") << ",\n";
			js << "    \"f_level\": " << result_v_f << ",\n";
		} else {
			js << "    \"achieved_frob\": null,\n";
			js << "    \"epsilon_passed\": false,\n";
			js << "    \"f_level\": null,\n";
		}
		js << "    \"method\": \"" << method << "\"\n";
		js << "  },\n";
		// unitary
		if (result_V_set) {
			js << "  \"unitary\": {\n";
			js << "    \"ring\": \"Z[zeta_9, 1/3]\",\n";
			js << "    \"basis\": \"1, zeta9, zeta9^2, zeta9^3, zeta9^4, zeta9^5\",\n";
			js << "    \"f\": " << f_common << ",\n";
			js << "    \"V\": [\n";
			for (int i = 0; i < 3; ++i){
				js << "      [" << emit_v_entry(i,0) << ", " << emit_v_entry(i,1) << ", " << emit_v_entry(i,2) << "]";
				if (i < 2) js << ",";
				js << "\n";
			}
			js << "    ]\n";
			js << "  },\n";
		} else {
			js << "  \"unitary\": null,\n";
		}
		// decomposition (HRSA Phase 3 only; null otherwise)
		if (result_dr_set) {
			js << "  \"decomposition\": {\n";
			js << "    \"N_D\": " << result_dr.D_count << ",\n";
			js << "    \"N_D_only\": " << result_dr.D_only << ",\n";
			js << "    \"N_R_only\": " << result_dr.R_only << ",\n";
			js << "    \"N_T_combined\": " << (10 * result_dr.D_only + 9 * result_dr.R_only) << ",\n";
			js << "    \"N_C\": null,\n";
			js << "    \"N_total\": null,\n";
			js << "    \"sde_chi_initial\": " << result_dr.sde_chi << ",\n";
			js << "    \"sde_chi_final\": " << result_dr.sde_chi_final << ",\n";
			js << "    \"syllables\": [";
			for (size_t k = 0; k < result_dr.steps.size(); ++k) {
				const GateStep& g = result_dr.steps[k];
				if (k) js << ", ";
				js << "{\"a0\": " << g.a0
				   << ", \"a1\": " << g.a1
				   << ", \"a2\": " << g.a2
				   << ", \"eps\": " << g.eps
				   << ", \"delta\": " << g.delta
				   << ", \"has_H\": " << (g.has_H ? "true" : "false")
				   << "}";
			}
			js << "]\n";
			js << "  },\n";
		} else {
			js << "  \"decomposition\": {\"N_D\": " << result_D
			   << ", \"N_C\": null, \"N_total\": null, \"sde_chi_initial\": null,"
			      " \"sde_chi_final\": null, \"syllables\": null},\n";
		}
		// sanity_checks
		js << "  \"sanity_checks\": {\n";
		js << "    \"unitarity_residual\": " << unitarity_residual << ",\n";
		js << "    \"frobenius_check_passed\": " << (found && result_frob < epsilon ? "true" : "false") << ",\n";
		js << "    \"decompose_roundtrip_passed\": null\n";
		js << "  },\n";
		// performance
		js << "  \"performance\": {\n";
		js << "    \"wall_seconds\": " << wall_seconds << ",\n";
		js << "    \"cpu_seconds\": " << wall_seconds << ",\n";
		js << "    \"max_rss_kb\": 0,\n";
		js << "    \"threads\": " << omp_get_max_threads() << ",\n";
		js << "    \"mpi_ranks\": null\n";
		js << "  },\n";
		// errors
		js << "  \"errors\": []\n";
		js << "}\n";
		js.close();
		cout << "JSON output written to " << json_path << endl;
	}

	return 0;
}
