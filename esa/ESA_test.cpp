#include<iostream>
#include "exhaustive_search.h"
#include<vector>
#include<array>
#include<cmath>
#include<complex>
#include<numeric>
#include<atomic>
#include<csignal>
#include<iomanip>
#include "cyclotomic_int9.h"
#include "Z9chi.h"
//#include "z3++.h"


std::atomic<bool> interrupted(false); // For clean exiting with Ctrl+C if execution takes too long

using namespace std;

int main(int argc, char* argv[]){
	
	signal(SIGINT, handleCtrlC);
	
	if(argc != 4){
		cout << "INVALID INPUT: Do ./ESA_test 'theta' 'epsilon' 'maximum f'." << endl;
	}
	
	double theta = atof(argv[1]);
	double epsilon = atof(argv[2]);
	int max_f = atoi(argv[3]);


// This code tests out the algorithm for a given choice of a diagonal.
// This particular example should give a non-diagonal solution with epsilon > 1.3,
// if one forces non-diagonal answers. 
/*

	ringZ9 x_1 = ringZ9(1,3) + ringZ9(3);

	
	array<ringZ9chi,9> output = testDiagonalESAWithSorting(x_1, x_1, x_1, theta, epsilon, max_f);

	for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				output[i + 3*j].print();
				cout << " ";
			}
			cout << "\n";
		}

	cout << "\n\n";
	*/
	
// This block of coode runs the ESA in full.	
	
	array<ringZ9chi,9> output2 = ESAWithSorting(theta, epsilon, max_f);
	
	// ---- Ring element representation ----
	cout << "\n--- Best approximation (ring representation) ---\n";
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			output2[i + 3*j].print();
			cout << " ";
		}
		cout << "\n";
	}

	// ---- Numerical matrix ----
	// Check whether a solution was actually found (a zero matrix signals failure).
	bool all_zero = true;
	for(int i = 0; i < 9; i++){
		complex<double> v = output2[i].toComplexDouble();
		if(abs(v) > 1e-12){ all_zero = false; break; }
	}

	if(all_zero){
		cout << "\nNo solution found within max_f = " << max_f
		     << " and epsilon = " << epsilon << ".\n";
		return 1;
	}

	cout << fixed << setprecision(8);
	cout << "\n--- Best approximation (numerical values) ---\n";
	for(int i = 0; i < 3; i++){
		cout << "[ ";
		for(int j = 0; j < 3; j++){
			complex<double> v = output2[i + 3*j].toComplexDouble();
			// print as (re, im) with consistent width
			cout << "(" << setw(12) << v.real()
			     << ", " << setw(12) << v.imag() << ")";
			if(j < 2) cout << "  ";
		}
		cout << " ]\n";
	}

	// ---- Frobenius error ||R^Z(theta) - V||_F ----
	// Target: R^Z_{(0,1)}(theta) = Diag(e^{-i*theta/2}, e^{+i*theta/2}, 1)
	// (canonical paper convention; matches HRSA and zeta9).
	complex<double> target[3][3] = {};
	target[0][0] = complex<double>(cos(theta/2), -sin(theta/2));
	target[1][1] = complex<double>(cos(theta/2),  sin(theta/2));
	target[2][2] = complex<double>(1.0, 0.0);

	double frob_sq = 0.0;
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			complex<double> diff = target[i][j] - output2[i + 3*j].toComplexDouble();
			frob_sq += norm(diff);   // norm() = |z|^2 for complex
		}
	}
	double frob = sqrt(frob_sq);

	cout << "\n--- Frobenius error ||R^Z(theta) - V||_F ---\n";
	cout << "theta        = " << theta  << "\n";
	cout << "epsilon      = " << epsilon << "\n";
	cout << "||R^Z - V||  = " << frob   << "\n";
	cout << "(epsilon check " << (frob < epsilon ? "PASSED" : "FAILED") << ")\n";

	return 0;
}



