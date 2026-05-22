#ifndef HOUSEHOLDER_H
#define HOUSEHOLDER_H
#include<iostream>
#include<cmath>
#include<array>
#include<numeric>
#include<complex>
#include<atomic>
#include<vector>
#include<unordered_map>
#include "cyclotomic_int9.h"
#include "Z9chi.h"

/**
* @file householder_search.h
* @brief Standalone functions for an Householder reflector search algorithm in 
* \f$ \mathbb{Z}[\zeta_9, \frac{1}{3}] \f$.
* 
* @author Chris H.
* @date March 2026
*******************************************************************************************************************/

/**
* @defgroup HouseholderSearch Householder Reflector Search Methods
* 
* @{ 
*******************************************************************************************************************/

/**
* @brief Performs Householder reflector search to approximate \f$ R_{(0,1)}^Z(\theta) \f$ up to \f$ \epsilon \f$ by
* finding a vector \f$ u \in \mathbb{Z}[\zeta_9, \frac{1}{3}]^3 \f$ close to \f$ [e^{i\theta/2}, -1, 0] \f$.
*
* We are trying to find a vector \f$ u := 3^{-f}[ x_1, x_2, x_3 ]^T \f$ with entries in \f$ \mathbb{Z}[\zeta_9, \frac{1}{3}] \f$ 
* differing from the vector \f$ [e^{i\theta/2}, -1, 0] \f$ in the Frobenius norm by at most \f$ \epsilon/(2c\sqrt{2}) \f$.
* We wish to choose the entries of \f$ u \f$ so that \f$ \| u \|^2 = 2 \f$. The Householder reflector is then obtained by
* taking \f$ H : = I_3 - u^T u \f$. Once this  \f$ H \f$ is found, we have that \f$ X_{(0,1)} H \f$ differs from the target
* unitary matrix \f$ R_{(0,1)}^Z(\theta) \f$ by \f$ \epsilon \f$ in the Frobenius norm.
* 
* Following along with the paper by Kalra et. al., we have a positive definite function \f$ q: \mathbb{Z}[\zeta_9] \to \mathbb{Z} \f$, 
* where \f$ x/3^f \f$ lies in a 3-vector of norm \f$ \sqrt{2} \f$ only if \f$ q(x) \leq 2\cdot 3^{2f} \f$. Since \f$ q \f$ is a
* positive-definite function on the coefficients of the coefficients (stored in the array 'element'), we are able to loop over
* all such \f$ x \f$ for a given \f$ f \f$. This is done in the function @ref entryEnumeration, where all candidates of \f$ x_i (i=1,2,3) \f$ are
* found in one function call.  
* 
* Once all candidates are found for each of the three entries, we loop over all possible combinations of \f$ x_1 \f$ and \f$ x_2 \f$.
* With a given choice of the first two entries, the third entry must satisfy 
* \f[
* |x_3|^2 = 2\cdot 3^{2f} - |x_1|^2 - |x_2|^2. 
* \f]
* Solving equations of the form \f$ |x|^2 = a \f$ for \f$ x \f$ appears to be difficult or this ring and seems to require us to solve
* a system of 6 quadratic forms in 18 variables. I do not believe that there is an efficient way to do this. This is why we collect a list of
* \f$ x_3 \f$ candidates in the std::vector 'lookup.' For each element of this std::vector we test whether the given \f$ x_3 \f$ satisfies the
* above equation.
* 
* If a valid \f$ x_3 \f$ is found that completes the vector, the \f$ epsilon \f$-condition is tested. If passed, the three entries are
* passed as a std::array. Otherwise, the next \f$ x_3 \f$ is tested. If all possible combinations of \f$ x_1, x_2, \f$ and \f$ x_3 \f$ 
* are exhausted, the value of f is incremented, and we start again.
* 
* @param theta The angle of the desired two-level Givens rotator \f$ R_{(0,1)}^Z(\theta) \f$.
* @param epsilon Desired level of precision of unitary to \f$ R_{(0,1)}^Z(\theta) \f$.
* @param max_f Largest value of f to be tested. Prevents program from running for an arbitrarily long time.
* @returns A std::array 'u' of 3 elements in \f$ \mathbb{Z}[\zeta_9, \frac{1}{3}] \f$, where \f$ X_{(0,1)}(I_3 - u^T u) \f$ approximates
* \f$ R_{(0,1)}^Z(\theta) \f$ within \f$ \epsilon \f$. 
*******************************************************************************************************************/
array<ringZ9chi,3> HRSA(double theta, double epsilon, int max_f, double c);

/**
* @brief Like HRSA, but collects up to max_solns valid vectors at the first
* successful f level, decomposes each into Clifford+D gates, and returns the
* one with the fewest D gates.
*
* @param theta    Target rotation angle.
* @param epsilon  Desired precision.
* @param max_f    Maximum f level to try.
* @param c        Contraction factor.
* @param max_solns Maximum number of candidate solutions to collect at the winning f.
* @returns The solution with the smallest D-gate count.
*************************************************************************************************************************/
array<ringZ9chi,3> HRSA_bestD(double theta, double epsilon, int max_f, double c, int max_solns = 50, int k3 = 1);

/**
* @brief For a given f value, this function fills the std::vectors x1_cands, x2_cands, and lookup with all 
* potential candidates for the entries \f$ x_1, x_2, \f$ and \f$ x_3 \f$ respectively, subject to the \f$ epsilon \f$-condition.
* 
* Analogously to the results of the Kalra et. al. paper, any choice of  \f$ x_1, x_2, \f$ and \f$ x_3 \f$ must 
* satisfy 
* \f[ 
* q(x_1) + q(x_2) + q(x_3) = 2\cdot 3^{2f}.
* \f]
* Since \f$ q(x) \f$ is positive-definite on the coefficients (the entries of the array 'element'), we may loop over 
* all possibilities \f$ x \f$, which necessarily satisfy \f$ q(x) \leq 2 \cdot 3^{2f} \f$.
* 
* For each \f$ x \f$ looped over, we check whether \f$ x \f$ is within \f$ \epsilon/(2c\sqrt{2}) \f$ of the three
* values: \f$ e^{i \theta/2}, -1, 0 \f$. Accordingly, \f$ x \f$ is placed into x1_cands, x2_cands, and/or x3_cands
* respectively.
* 
* After looping through all possibilities, the entries of each of the 3 vectors are sorted by how close they are to
* their respective target values.
* 
* @param x1_cands Vector holding all possible choices of \f$ x_1 \f$ for given f.
* @param x2_cands Vector holding all possible choices of \f$ x_2 \f$ for given f.
* @param lookup Hash map (keyed by quad() value) holding all possible choices of \f$ x_3 \f$ for given f. Will be used in @ref HRSA to find
* solutions to the equation \f$ |x_3|^2 = 2\cdot 3^{2f} - |x_1|^2 - |x_2|^2. \f$
* @param theta The angle of the desired two-level Givens rotator \f$ R_{(0,1)}^Z(\theta) \f$.
* @param epsilon Desired level of precision of unitary to \f$ R_{(0,1)}^Z(\theta) \f$.
* @param f Power of three in denominator.
* @param c Correction factor from original paper.
*******************************************************************************************************************/
void entryEnumeration(	std::vector<ringZ9>& x1_cands,
						std::vector<ringZ9>& x2_cands, 
						std::unordered_map<int,std::vector<ringZ9>>& lookup,
						double theta, double epsilon, int f, double c);

/**
* @brief Function called once 3-vector of norm \f$ \sqrt{2} \f$ is found. Tests if vector is within \f$ \epsilon/(2c\sqrt{2}) \f$
* of target vector \f$ [e^{i\theta/2}, -1, 0] \f$ in the Frobenius norm. 
*
* This only tests the \f$ epsilon \f$-condition between the 3-vectors and NOT the unitaries. Might want to code that
* in soon. 
* 
* @param u_1 First entry of approximation vector u.
* @param u_2 Second entry of approximation vector u.
* @param u_3 Third entry of approximation vector u.
* @param f Power of three in denominator.
* @param angle_dir This is \f$ e^{i \theta/2} \f$ stored as a complex<double>.
* @param epsilon Desired level of precision of unitary to \f$ R_{(0,1)}^Z(\theta) \f$.
* @param c Correction factor from original paper.
*******************************************************************************************************************/
bool epsTest(ringZ9chi u_1, ringZ9chi u_2, ringZ9chi u_3, int f, std::complex<double> angle_dir, double epsilon, double c);

/**
* @brief Computes 3^n using int arithmetic. Alternative to using std::pow which returns doubles.
************************************************************************************************************************/
int three_power(int n);
/** @} */

/**
* @brief Computes the actual Frobenius distance \\f$ \\| X_{(0,1)}(I_3 - u^T u) - R_{(0,1)}^Z(\\theta) \\|_F \\f$
* and prints it alongside the target epsilon.
*
* This is the true end-to-end check: it constructs the 3×3 Householder reflector
* \\f$ H = I_3 - u^T u \\f$ from the found vector \\f$ u \\f$, applies the fixed correction
* matrix \\f$ X_{(0,1)} = \\begin{pmatrix} 0&1&0 \\\\ 1&0&0 \\\\ 0&0&1 \\end{pmatrix} \\f$ (the
* (0,1)-SWAP), and measures how far \\f$ X_{(0,1)} H \\f$ is from the target unitary
* \\f$ R_{(0,1)}^Z(\\theta) = \\mathrm{Diag}(e^{-i\\theta/2},\\, e^{i\\theta/2},\\, 1) \\f$
* in the Frobenius norm.
*
* The vector-level epsTest guarantees this Frobenius distance is less than epsilon (given
* \\f$ c=1 \\f$), but the actual value can be substantially smaller. Printing it lets you
* see the true quality of the approximation rather than just the proxy condition.
*
* @param u     The 3-vector returned by HRSA.
* @param theta The target rotation angle (the EXTERNAL theta, before negation).
* @param epsilon The target precision, printed alongside the result for comparison.
*************************************************************************************************************************/
void matrixFrobeniusCheck(const std::array<ringZ9chi,3>& u, double theta, double epsilon);

/**
* @brief This code allows us to exit gracefully when Ctrl+C is used during runtime.
* 
* Since we are dealing with large data structures, using Ctrl+C during runtime might result in memory leaks since the
* execution is abruptly halted. Now, when Ctrl+C is called, the program is allowed to terminate, thus allowing
* any destructors to be called for the vectors we've used.
*************************************************************************************************************************/
void handleCtrlC(int sig);

#endif
