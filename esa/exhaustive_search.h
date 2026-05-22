#ifndef EXHAUSTIVE_SEARCH
#define EXHAUSTIVE_SEARCH
#include<iostream>
#include<forward_list>
#include<vector>
#include<array>
#include<algorithm>
#include<cmath>
#include<complex>
#include<numeric>
#include<atomic>
#include<csignal>
#include<unordered_map>
#include<omp.h>
#include "cyclotomic_int9.h"
#include "Z9chi.h"


using namespace std;

/**
* @file exhaustive_search.h
* @brief Standalone functions for an exhaustive search algorithm in 
* \f$ \mathbb{Z}[\zeta_9, \frac{1}{3}] \f$.
* 
* @author Chris H.
* @date June 2025
*******************************************************************************************************************/

/**
* @defgroup ExhaustiveSearch Exhaustive Search Methods
* 
* This file contains @ref ESA (exhaustive search algorithm) and all of the component functions
* it calls. The algorithm is a modification of the exhaustive search algorithm described <a href="https://arxiv.org/abs/2503.20203">here</a>,
* where we instead work over the Clifford \f$ + D \f$ froup as opposed to the Clifford \f$ + R \f$ group. Specifically,
* the algorithm tries to approximate the matrix \f$ R_{(0,1)}^Z(\theta) \f$ up to a precision of \f$ \epsilon \f$
* under the Frobenius norm with an element of the unitary group \f$ U(\mathbb{Z}[\zeta_9, \frac{1}{3}]) \f$. 
* Here, \f$ R_{(0,1)}^Z(\theta) = \mbox{Diag}(e^{-i\theta/2}, e^{i\theta/2},1 \f$. The algorithm follows the same general process
* outlined in Algorithm 1 of the original paper. Our analog of the ENUMERATECANDIDATES routine is the
* @ref fullDiagEnumeration, and for COMPLETEUNITARY, @ref exhaustiveCompleteUnitary. 
*
* That this algorithm is truly exhaustive relies primarily on Theorem 4.3 <a href="https://arxiv.org/abs/2311.08696">here</a>.
* In particular, they work with the ring \f$ \mathbb{Z}[\zeta_9, (1 - \zeta)^{-1}] \f$. This ring is really the same
* as \f$ \mathbb{Z}[\zeta_9, \frac{1}{3}] \f$. To see this, note that \f$(1 - \zeta_9)^9 \f$ is divisible by 3 when expressed
* in the natural \f$ \mathbb{Z} \f$-basis \f$ \{ \zeta_9^i \}_{i=0}^5 \f$. Thus
* \f[
* 1/3 = \frac{(1 - \zeta_9)^9 / 3}{(1-\zeta_9)^9} \in \mathbb{Z}[\zeta_9, (1 - \zeta_9)^{-1}].
* \f]
* Likewise, taking the derivative on both sides of the formula for a finite geometric squence, one can obtain
* \f[
*  (1 - \zeta_9)^{-1} = \frac{-1}{3^2} \sum_{i=1}^{8} ix^i \in \mathbb{Z}[\zeta_9, \frac{1}{3}]
* \f]
* Knowing that these two rings are the same, we will refer to this ring as \f$ R_D \f$. Now, \f$ R_D \f$ is dense in
* the complex plane, which poses a challenge enumerating ALL candidates for a diagonal element. However, Theorem 4.3 mentioned
* above states that there are finitely many elements of \f$ R_D \f$ which arise as entries of a unit vector for a given \f$ f \f$.
* (Note that replacing their \f$ \chi^f \f$ with \f$ 3^f \f$ makes no fundamental difference except for the specific values of 
* \f$ A, B, \f$ and \f$ C \f$ in their quadratic forms.) The relations given in the theorem also provide us with a positive
* definite quadratic form \f$ q \f$ with an upper bound given in terms of \f$ f \f$. This allows us to obtain ALL diagonal
* candidates by means of a nested for-loop (see @ref fullDiagEnumeration), and we can also form a lookup table for candidates 
* belowt the main diagonal. Once a main diagonal and the lower entries determined, the entries above the diagonal can be found algebraically (see 
* @ref fillUpper).
* 
* @note We store our diagonal candidates and lookup table in a <a href="https://en.cppreference.com/w/cpp/container/forward_list.html">std::forward_list</a>.
* I prefer these to std::vectors since vectors are stored in one contiguous chunk of memory, whereas forward_lists do not require this. Initially,
* I implemented this with vectors, but this seems as though it could lead to memory issues as f is increased and the lookup table gets larger. A downside
* is that although we no longer require large contiguous chunks of memory, we ultimately use more memory as each element of the list also has a pointer
* to the next element. Also, as forward_lists are somewhat more primitive, we can only iterate through these in one direction, and we cannot directly
* compute the size.
* @{ 
*******************************************************************************************************************/

/**
* @brief Performs a exhaustive search to approximate \f$ R_{(0,1)}^Z(\theta) \f$ with an 
* element of \f$ \mbox{U}(3, \mathbb{Z}[\zeta_9, \frac{1}{3}]) \f$.
*
* Our code somewhat closely resembles the pseudocode for the ESA (Algorithm 1) in the original paper.
* However, since we take our denominators to be powers of 3 instead of \f$ \chi = 1 + 2\zeta_3 \f$, there 
* is no change-of-variable as was done in line 2 of Algorithm 1.
* The algorithm starts with f = 0, but if no satisfactory unitary is found, f is incremented, and the search process
* repeats. After the algorithm terminates will have a \f$ 3 \times 3 \f$ matrix of the form 
*
* \f[
*  3^{-f} \cdot \begin{bmatrix} x_1 & y_1 & z_1 \\ x_2 & y_2 & z_2 \\ z_3 & y_3 & z_3 \end{bmatrix}
* \f]
* where the main diagonal and lower entries will be elements of \f$ \mathbb{Z}[zeta_9] \f$.
*
* @param theta The angle of the desired two-level Givens rotator \f$ R_{(0,1)}^Z(\theta) \f$.
* @param epsilon Desired level of precision.
* @param max_f Largest value of f to be tested. Prevents Program from running for an arbitrarily long time.
* @returns A \f$ 3 \times 3 \f$ matrix \f$ V \in \mbox{U}(3, \mathbb{Z}[\zeta_9, \frac{1}{3}]) \f$ such that 
* \f$ \| R_{(0,1)}^Z(\theta) - V \| < \epsilon \f$ under the Frobenius norm. This matrix \f$ V \f$ 
* is stored as a std::array with the first three entries corresponding to the first column, the next three
* entries corresponding to the second column, etc. 
*******************************************************************************************************************/
array<ringZ9chi,9> ESA(double theta, double epsilon, int max_f);



/**
* @brief Unlike my previous enumeration methods, this enumerates ALL diagonal candidates for a given f, theta, and epsilon.
* 
* We take advantage of the relations found in Theorem 4.3 of <a href="https://arxiv.org/abs/2311.08696">Synthesis and Arithmetic of Single Qutrit Gates</a>
* Going through the \f$ \S 4 \f$ replacing \f$ \chi^f \f$ with \f$ 3^f \f$, we set 
* \f[
*  q = \frac{1}{4}( 2a_0 - a_3)^2 + 3a_3^2 + (2a_1 - a_4)^2 + 3a_4^2 + (2a_2 - a_5)^2 + 3a_5^2
* \f]
* If the \f$ a_i \f$'s represent the coefficients of a reduced element \f$ x \in \mathbb{Z}[\zeta_9] \f$, then  
* Theorem 4.3 implies \f$ q(x) + q(y) + q(z) = 3^{2f} \f$ where \f$ [x,y,z]^T \f$ is a unit vector. Since the form
* \f$ q \f$ is positive definite, this implies \f$ q(x) \leq 3^{2f} \f$, and further, we may loop through all 
* possible coefficients of \f$ x \f$.
* 
* The value of this is that, even though our desired region contains infinitely many elements of \f$ \mathbb{Z}[\zeta_9] \f$,
* only finitely many of them can be entries of a unit vector.
* 
* @note I have included a condition that prevents elements divisible by 3 from being included. Since 
* our matrix entries will have the same \f$ \mbox{sde}_\chi \f$, we do not need to check these as these will have been checked
* on a previous iteraton.
* 
* @param candidates A forward_list which will be filled with all candidates for a diagonal entry in a given region.
* @param theta The angle of the desired two-level Givens rotator \f$ R_{(0,1)}^Z(\theta) \f$.
* @param epsilon Desired level of precision.
* @param f Power of three in denominator.
************************************************************************************************************************/
void fullDiagEnumeration(array<vector<ringZ9>,3>& candidates, double theta, double epsilon, int f,
                         size_t max_candidates = 500);


/**
* @brief For \f$ z \in \mathbb{Z}[\zeta_9] \f$ , computes the real part of \f$ z \cdot e^{i\alpha} \f$.
*
* This is used in @ref fullDiagEnumeration since a given diagonal element \f$ x \f$ must satisfy
*  \f$ \mbox{Re}(z \cdot e^{i\alpha}) \geq 3^{f}(1 - \epsilon^2/2) \f$.
* 
* @param test A candidate for some diagonal entry.
* @param angle Plays the role of \f$ \alpha \f$ above.
************************************************************************************************************************/
double planeSide(ringZ9 test, double angle);


/**
* @brief Loops over forward_list to find the minimum value taken by form \f$ q \f$
* 
* Given a list of diagonal candidates, we want to find the minimum value taken by \f$ q \f$ among
* all possible candidates for either \f$ x_1 \f$ or \f$ z_3 \f$. This will help us restrict our search  
* (at least marginally) for candidates of \f$ x_3 \f$ in @ref fullX3Enumeration and hopefully shorten
* search times in @ref exhaustiveCompleteUnitary.
* 
* @param vec A forward_list of diagonal candidates for either \f$ x_1 \f$ or \f$ z_3 \f$.
* @param f Power of three in denominator.
* 
* @returns Smallest values taken by form \f$ q \f$ out of all ring elements in vec.
************************************************************************************************************************/
int findMinQ(vector<ringZ9>& vec, int f);


/**
* @brief Computes a table of all possible values for \f$ x_3 \f$ that will also be used as a lookup table for finding all
* possible values of \f$ x_2 \f$ and \f$ y_3 \f$.
* 
* We use the same principles as in @ref fullDiagEnumeration but since \f$ x_1 \f$ and \f$ z_3 \f$ are predetermined at
* the point this fucntion is called, we can restrict the search using the parameter M. We have that
* \f[
* q(x_3) + q(x_1) \leq 3^{2f} \; \mbox{and} \; q(x_3) + q(z_3) \leq 3^{2f},
* \f]
* so M (which is found by @ref findMinQ ) is taken to be the minimum of all possible values of \f$ q \f$ ranging
* over candidates of \f$ x_1 \f$ and \f$ z_3 \f$. Thus, we loop over all possibilities of instance of @ref ringZ9 whose
* evaluation in \f$ q \f$ is less than \f$ 3^{2f} - M \f$.
* 
* @note I have included a condition that prevents elements divisible by 3 from being included. Since 
* our matrix entries will have the same \f$ \mbox{sde}_\chi \f$, we do not need to check these as these will have been checked
* on a previous iteraton.
* 
* @param candidates A forward_list which will be filled with all candidates for \f$ x_3 \f$. 
* @param f Power of three in denominator.
* @param M The minimum value of \f$ q \f$ taken by any candidate for either \f$ x_1 \f$ or \f$ z_3 \f$.
* @param epsilon Desired level of precision
***********************************************************************************************************************/
void fullX3Enumeration(vector<ringZ9>& candidates, int f, int M, double epsilon);



/**
* @brief Tests whether a given triple of diagonal candidates is possible for a \f$ 3 \times 3 \f$ unitary matrix.
*
* Before running @ref exhaustiveCompleteUnitary, it is wise to test whether a given triple of diagonal candidates
* is possible for a unitary matrix of the desired precision. Equation (27) in the <a href="https://arxiv.org/abs/2503.20203">original paper</a>
* gives a necessary and sufficient condition, which is checked against the arguments.
*
* @note This used to check that the epsilon choice hasn't already been overshot by the given diagonal but the need
* for this was superseded my more efficient checks in the main body @ref ESA.
*
* @param x_1 Candidate for the entry \f$ x_1 \f$
* @param y_2 Candidate for the entry \f$ y_2 \f$ 
* @param z_3 Candidate for the entry \f$ z_3 \f$
* @param f_pow Three to the power of current f as a double
* @returns True if the entries form a possible diagonal of a \f$ 3 \times 3 \f$ unitary matrix,
* and if the error of \f$ \mbox{Diag}(x_1, y_2 ,z_3) \f$ from the desired Givens rotator does
* not exceed epsilon.
*******************************************************************************************************************/
bool unitaryDiagCheck(ringZ9 x_1, ringZ9 y_2, ringZ9 z_3, double f_pow);


/**
* @brief Given a choice of diagonal candidates, this routine attempts to complete the remaining entries of the unitary.
* 
* The algorithm loops through lookup looking for \f$ x_3 \f$ candidates. For a given candidate, lookup is looped through
* once again looking for candidates of \f$ x_2 \f$ and \f$ y_3 \f$ which make the first column and third row unit vectors.
* If and when valid candidates are found, the routine calls @ref fillUpper which tries to fill entries aove the main diagonal. 
* If this succeeds, the matrix is checked against the epsilon condition and to see whether the matrix is unitary. If no
* choice for \f$ x_3 \f$ in lookup is suitable, the zero matrix is returned.
* 
* @param x_1 Candidate for the entry \f$ x_1 \f$
* @param y_2 Candidate for the entry \f$ y_2 \f$ 
* @param z_3 Candidate for the entry \f$ z_3 \f$
* @param theta The angle of the desired two-level Givens rotator \f$ R_{(0,1)}^Z(\theta) \f$.
* @param epsilon Desired level of precision
* @param f Power of three in denominator.
* @returns If a satisfactory unitary is found, the matrix itself is returned as a std::array with the
* exact values of the found entries. Again, the first three entries of the array hold the first column, etc.
* Otherwise, if no satisfactory unitary was found, the zero matrix is returned, and @ref ESA continues
* to the next iteration.
************************************************************************************************************************/
// Fix G.1 (2026-05): added x1conj, x3conj, z3conj as precomputed-conjugate
// parameters so callers can hoist these out of the inner loops where
// (x_1, x_3, z_3) are loop-invariant.  fillUpper used to call .complexConj()
// on these per (x_2, y_3) candidate pair; now they're computed once per
// (x_1, x_3, z_3) instead.
pair<array<ringZ9chi,9>,bool> exhaustiveCompleteUnitary(ringZ9 x_1, ringZ9 y_2, ringZ9 z_3, ringZ9 x_3,
                                                       const vector<ringZ9>& x2_cands, const vector<ringZ9>& y3_cands,
                                                       double theta, double epsilon, int f,
                                                       const ringZ9& x1conj, const ringZ9& x3conj, const ringZ9& z3conj);


/**
* @brief Fills entries above diagonal (if possible) once all other entries are determined.
*
* Once the main diagonal and the lower entries are selected, IF the matrix can be completed
* into a unitary matrix, the three entries above the main diagonal are uniquely determined by 
* the unitarity of the matrix, namely since the columns (resp. the rows) are mutually orthogonal.
* For instance, \f$ y_1 \f$ is found by solving the below equation: 
*
* \f[ 
* \bar{x_1} \cdot y_1 = -(\bar{x_2} \cdot y_2 + \bar{x_3} \cdot y_3 )
* \f]
* A similar equation is formed for solving \f$ z_2 \f$ and \f$ z_1 \f$.
* 
* @warning We assume that the parameters x_1 and z_3 are zero. This should not be an issue since
* @ref ESA shrinks epsilon to 0.99 if epsilon is greater than 1, but otherwise, the program is 
* likely to throw a floating point exception.
*
* @param unitary A std::array which holds the values so far determined. Namely, the main diagonal
* and the three entries below it.
* @param x2_cand Candidate for \f$ x_2 \f$.
* @param y3_cand Candidate for \f$ y_3 \f$.
* @param f Power of three in denominator.
* @returns A std::pair holding on the right a bool which is true iff the algorithm was able to
* successfully each of the remaining unitary entries. If the right is true, the left holds a std::array
* with all entries initialized as instances of @ref ringZ9chi. At this point it remains to check that the
* given matrix is unitary and satisfies the epsilon bound. 
*******************************************************************************************************************/
// Fix G.1 (2026-05): see comment at exhaustiveCompleteUnitary above.
pair<array<ringZ9chi, 9>, bool> fillUpper(array<ringZ9,9> unitary, ringZ9 x2_cand, ringZ9 y3_cand, int f,
                                          const ringZ9& x1conj, const ringZ9& x3conj, const ringZ9& z3conj);


/**
* @brief If possible, this solves equations of the form \f$ ax = b \f$ for \f$ x \f$ in the ring \f$ \mathbb{Z}[\zeta_9] \f$.
*
* Although the rings involved are Euclidean domains, there does not seem to be a Euclidean 
* algorithm that is easily implementable in code to do the necessary division. However, what we
* can do instead is multply the numerator and denominator \f$ \frac{a}{b} \f$ by the @ref ringZ9::partialFieldNorm of \f$ b \f$,
* which is \f$ \prod_{\sigma \in G - \{1\}} \sigma(b) \f$ where \f$ G = \mbox{Gal}(\mathbb{Q}(\zeta_9)) \f$.
* This makes the denominator into the field norm which is an integer, say \f$ N \f$. Now, we need only test whether the 
* numerator times the partial field norm is divisible by \f$ N \f$, and since we may assume that this new numerator is 
* written in the natural basis, this amounts to checking that each coefficient is divisible by \f$ N \f$.  
*
* @note From the results of <a href="https://arxiv.org/abs/2311.08696">this papera</a>, we can assume that the solution
* found has the same sde as that of numer and denom. Thus, our result will not require a larger value of f and we may
* return our answer as an instance of @ref ringZ9.
* 
* @warning The field norm of an element having large absolute value can have a very large norm, so there is a strong risk of integer
* overflow for values of \f$ f \f$ greater than two.
* 
* @param numer Plays the role of \f$ b \f$ above.
* @param denom Plays the role of \f$ a \f$ above.
* 
* @returns A std::pair where the right is true iff the equation was solved. If the right is true, then
* the left returns the solution \f$ x \f$ as a @ref ringZ9. 
*******************************************************************************************************************/
pair<ringZ9,bool> solveSystem(ringZ9 numer, ringZ9 denom);

/**
* @brief Checks whether a given \f$ 3 \times 3 \f$ matrix is unitary/
*
* This verifies that the columns are oairwise orthogonal unit vectors. We need only check whether 
* the second and third columns are unit vectors, since the first is a unit vector by construction. Likewise, 
* we only need to verify that the first and third columns are orthogonal as well as the second and third columns.
* 
*
* @param unitary A std::array viewed as a matrix which is to be checked,
*******************************************************************************************************************/
bool isUnitary(array<ringZ9chi,9> unitary);

/**
* @brief Verifies that the found unitary matrix sufficiently approximates \f$ R_{(0,1)}^Z(\theta) \f$
*
* Before @ref exhaustiveCompleteUnitary terminates with a solution, we need to test the following condition:
* \f[
*	\| R_{(0,1)}^Z(\theta) - V \| < \epsilon
* \f]
* where \f$ V \f$ is the found unitary.
*
* @param unitary Found unitary matrix stored in a std::array.
* @param theta The angle of the desired two-level Givens rotator \f$ R_{(0,1)}^Z(\theta) \f$.
* @param epsilon Desired level of precision.
* @returns True if epsilon condition is satisfied and false otherwise.
************************************************************************************************************************/
bool checkEpsilon(array<ringZ9chi,9> unitary, double theta, double epsilon);

/**
* @brief Takes a vector of candidates and sorts them into an array by \f$ \mbox{sde}_\chi \f$
* 
* Since @ref fullDiagEnumeration and @ref fullX3Enumeration discarded all candidates with \f$ \mbox{sde}_\chi  > 5\f$,
* (equivalent to divisibility by 3), there are only six possibilities for the sde: 0 through 5. The sde in terms of chi
* is computed using the so-called derivative mod \f$ p \f$ in section 3 of the Kalra paper (see @ref ringZ9::sdeChi).
* 
* @note When the lookup table is passed as an argument, we must also add the zero element to each of the six resulting
* vectors to prevent diagonal appproximations from being excluded. 
***********************************************************************************************************************/
void sortBySdeChi(vector<ringZ9> &unsorted, array<vector<ringZ9>,6> &sorted);

/**
* @brief Computes 3^n using int arithmetic. Alternative to using std::pow which returns doubles.
************************************************************************************************************************/
int three_power(int n);

/**
* @brief Performs ESA, but sorts diagonal and x_3 candidates by \f$ \mbox{sde}_\chi \f$ for faster runtime.
* 
* Lemma 5.1 in the Kalra paper states that the entries in our matrices must all have the same sde in terms of chi. 
* Since @ref fullDiagEnumeration and @ref fullX3Enumeration discarded all candidates with \f$ \mbox{sde}_\chi  > 5\f$,
* (equivalent to divisibility by 3), we sort our candidates by their sde's in terms of chi, and try to complete
* the unitary ONLY with candidates whose sde matches. This provides a speedup.
* 
************************************************************************************************************************/
array<ringZ9chi, 9> ESAWithSorting(double theta, double epsilon, int max_f);
/** @} */


/**
* @defgroup DebugTools ESA Debugging Tools
* 
* These methods are not part of the exhaustive search algorithm proper but may be used to test for algorithm success in
* specific cases.
* 
* @{
************************************************************************************************************************/
/**
* @brief Performs an iteration of the ESA for a specific diagonal. For debugging purposes. 
* 
* For this routine, you may pass a diagonal for which you know a solution should exist for the given theta, epsilon, and
* for some f less than max_f. If this function does not return a valid answer, then this can indicate the existence of bug(s) 
* somewhere in the @ref ESA or in one of the functions it calls.
************************************************************************************************************************/
array<ringZ9chi, 9> testDiagonalESA(ringZ9 a1, ringZ9 a2, ringZ9 a3, double theta, double epsilon, int max_f);

/**
* @brief Like @ref testDiagonalESA, but performs ESA with new \f$ \mbox{sde}_\chi \f$ sorting technique. 
* 
***********************************************************************************************************************/
array<ringZ9chi, 9> testDiagonalESAWithSorting(ringZ9 a1, ringZ9 a2, ringZ9 a3, double theta, double epsilon, int max_f);

/**
* @brief For a given choice of theta, epsilon, and f, this tests whether @ref fullDiagEnumeration was able to find the 
* test diagonal passed (x_1, y_2, z_3).
* 
* This method is used in @ref testDiagonalESA. This essentially can be used to make sure that the ESA al
*************************************************************************************************************************/
bool foundInDiagTable(ringZ9 x_1, ringZ9 y_2, ringZ9 z_3, double theta, double epsilon, int f);
/** @} */


/**
* @brief This code allows us to exit gracefully when Ctrl+C is used during runtime.
* 
* Since we are dealing with large data structures, using Ctrl+C during runtime might result in memory leaks since the
* execution is abruptly halted. Now, when Ctrl+C is called, the program is allowed to terminate, thus allowing
* any destructors to be called for the vectors we've used.
*************************************************************************************************************************/
void handleCtrlC(int sig);

#endif
