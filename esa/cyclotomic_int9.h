#ifndef RINGZ9_H
#define RINGZ9_H

#include<iostream>
#include<cmath>
#include<array>
#include<numeric>
#include<complex>

# define M_PI           3.14159265358979323846

using namespace std;

/**
* @class ringZ9 
* @brief Instances of the class represent elements of the ring \f$ \mathbb{Z}[\zeta_9] \f$ where
* \f$ \zeta_9 \f$ is the primitive ninth root of unity \f$ e^{2\pi/9} \f$.
*
* This class allows for basic arithmetic with elements of \f$ \mathbb{Z}[\zeta_9] \f$ and provides
* member functions that do some basic number-theoretic operations. The ring itself consists of all
* possible sums of ninth roots of unity of the form:
* \f[
*  \sum_{i=0}^8 a_i \zeta_9^i
* \f]
* for \f$ a_i \in \mathbb{Z} \f$.
*
* @author Chris H.
* @date June 2025
**********************************************************************************************************************/
class ringZ9{
	
	private:
	/** 
	* @brief Array for the coefficients on each of the nine rou terms.
	*
	* Every element of the ring may be expressed as a sum of ninth roots of unity. In particular, the 
	* \f$ i \f$-th entry of the array stores the integer coefficient on the \f$ \zeta_9^i \f$ term.
	*******************************************************************************************************************/
	int element[9];
	
	public:
	

	/* CONSTRUCTORS */
	/** 
	* @defgroup ringZ9Constructors ringZ9 Constructors
	* @brief Various constructors for an element
	* @{
	*******************************************************************************************************************/
	
	ringZ9(int arr[9]); ///< Defines ring element explicitly by copying argument 'arr' to @ref ringZ9::element.
	
	ringZ9(int n); ///< Sets element equal to a rational integer \f$ n \f$.
	
	/**
	* @brief Defines an element of the ring consisting of a single term, \f$ \mbox{coeff} \cdot \zeta_9^{\mbox{term}} \f$.
	* @param coeff This is the integer coefficient or the term.
	* @param term This will be the exponent on the rou to which coeff is multiplied.
	*******************************************************************************************************************/
	ringZ9(int coeff, int term);

	ringZ9(); ///< Sets ring element equal to the zero element.
	/** @} */ 
	
	/* INFO */
	/** 
	* @defgroup ringZ9Info ringZ9 Info
	* @brief Member functions returning specific information on ring element.
	* @{
	*******************************************************************************************************************/
	int getTerm(int i) const; ///< Gets integer coefficient on the \f$\zeta_9^i\f$ term.
	
	/** 
	* @brief Will return @ref ringZ9::element as a std::array.
	*
	* See <a href="https://en.cppreference.com/w/cpp/container/array.html?ref=blog.crespum.eu">here</a> for 
	* more info on std::array container.
	*******************************************************************************************************************/
	array<int, 6> getStdArray() const; 
	/** @} */
	
	/* MUTATORS */
	/**
	* @defgroup ringZ9Mutators ringZ9 Mutators
	* @brief Member functions which have authority to modify the ring element itself.
	* 
	* @note Scalar multiplication and division are overloaded below. For cleaner code, those should be used instead.
	* @{
	*******************************************************************************************************************/
	/**
	* @brief Rewrites terms of @ref element into a canonical basis.
	*
	* This ring has a \f$\mathbb{Z}\f$-basis given by \f$\zeta_9^i\f$ for \f$i=0,1,\ldots,5\f$. Thus, any
	* has a unique representation using only the first six entries of @ref ringZ9::element with the last three
	* entries being zero. The function does this by taking advantage of the relations
	* \f[
	*		1 + \zeta_9^3 + \zeta_9^6 = 0
	* \f] \f[
	*		\zeta_9 + \zeta_9^4 + \zeta_9^7 = 0
	* \f] \f[
	*		\zeta_9^2 + \zeta_9^5 + \zeta_9^8 = 0
	* \f]
	* 
	* @warning Most of the member functions in this class assume that a given ring element is in its
	* reduced form. The constructors which have a possibility to initialize the last three entries
	* will call reduce() automatically. Any member function (such as @ref GaloisAut) 
	* or operation (such as *) are likely to produce non-reduced instances and must call reduce()
	* before terminating.
	*******************************************************************************************************************/
	void reduce();
	void scalar_mult(int scalar); ///< Multiplies each term by 'scalar.'
	void scalar_div(int scalar); ///< Performs integer division by 'scalar' on each term. Should only be used if each coefficient is divisble by 'scalar.'
	/** @} */
	
	/* OPERATIONS */
	/**
	* @defgroup ringZ9BasicOperations ringZ9 Basic Operations
	* @brief Operations of addition, subtraction, multiplication, and assignment defined in the usual way.
	* @{
	*******************************************************************************************************************/
	ringZ9 operator+(const ringZ9& right) const; ///< Ring addition
	ringZ9 operator-(const ringZ9& right) const; ///< Ring subtraction
	ringZ9 operator*(const ringZ9& right) const; ///< Multiples two ring elements using distributive property and returns product after calling @ref ringZ9::reduce
	ringZ9 operator=(const ringZ9& right); ///< Ring assignment
	/** @} */
	
	/**
	* @defgroup ringZ9ScalarOps ringZ9 Scalar Operations
	* @brief Overloads scalar multiplaction by scalars.
	* @note The syntax requires that the scalars act from the right.
	* @{
	*******************************************************************************************************************/
	ringZ9 operator*(const int& right_scalar) const; ///< Scalar multiplication
	ringZ9 operator/(const int& right_scalar) const; ///< Scalar (integer) division
	/** @} */
	
	/**
	* @defgroup ringZ9Equality ringZ9 Equality
	* @brief Since we may assume both sides are reduced, it suffices to check whether 
	* both @ref ringZ9::element arrays are identical to verify equality.
	* @{
	*******************************************************************************************************************/
	bool operator==(const ringZ9& right) const; ///< Is equal to
	bool operator!=(const ringZ9& right) const; ///< Is not equal to
	/** @} */
	
	/* TYPE CONVERSIONS */
	/**
	* @defgroup ringZ9TypeConversions ringZ9 Type Conversions
	* @{ 
	*******************************************************************************************************************/
	double real_part() const; ///< Real part 
	double imag_part() const; ///< Imaginary part
	double complexArg() const; ///< Returns argument of ring element viewed as complex number.
	double abs_val() const; ///< Absolute value
	double abs_val_sq() const; ///< Absolute value squared
	complex<double> toComplexDouble() const; ///< Converts to std::complex<double>
	/** @} */
	
	/* NUMBER-THEORETIC */
	/**
	* @defgroup ringZ9NumberTheoretic ringZ9 Number-theoretic
	* @brief Some basic number-theretic routines.
	* @{
	*******************************************************************************************************************/
	/**
	* @brief Computes Galois conjugate of given instance.
	*
	* The Galois group of \f$ \mathbb{Q}(\zeta_9) \f$ is the cyclic group of order six. The Galois group
	* consists of field automorphisms from \f$ \mathbb{Q}(\zeta_9) \f$. By 'automorphism,' we mean abort
	* bijective function form the field to itself which a) distributes over addition, multiplication, subtraction,
	* and division, and b) maps 0 and 1 to themselves. Because of these properties, any Galois 
	* automorphism is defined uniquely by where it maps \f$ \zeta_9 \f$ (which must also get mapped
	* to a root of unity). In particular, the Galois automorphisms in this case exponentiate the root of unity
	* in each term to some power \f$ k \f$, where gcd(k,9) = 1. That is, each term \f$ a_i \zeta_9^i \f$
	* becomes \f$ a_i \zeta_9^{i*k} \f$.
	*
	* @param k This integer (if relatively prime to 9) defines a Galois automorphism, and is used to 
	* exponentiate the rou in each term.
	* @warning The parameter k defines a true Galois automorphism iff gcd(k,9) = 1. The member
	* function does NOT	test this gcd condition.
	*******************************************************************************************************************/
	ringZ9 GaloisAut(int k) const;
	
	/**
	* @brief Computes complex conjugate by calling GaloisAut(-1)
	*
	* Complex conjugation is an example of a Galois automorphism where the exponent on each rou
	* is made negative.
	*******************************************************************************************************************/
	ringZ9 complexConj() const;
	
	/**
	* @brief Computes the product of all Galois conjugates (which is guaranteed to be an integer).
	*
	* See <a href="https://en.wikipedia.org/wiki/Field_norm">here</a> for some info and general properties of 
	* the field norm.
	*
	* @note As a consequence of the Fundamental Theorem of Galois Theory, in certain fields (such 
	* as \f$ \mathbb{Q}(\zeta_9) \f$, a field element is rational iff it is fixed by every Galois
	* automorphism, since Galois automorpihsms distribute over multiplication, any Galois 
	* automorphism will simply permute the factors of the field norm, meaning the field norm must be
	* an integer, so converting to int is always valid.
	*******************************************************************************************************************/
	int fieldNorm() const;
	
	/**
	* @brief Computes the sum of all Galois conjugates (which is guaranteed to be an integer)
	*
	* See <a href="https://en.wikipedia.org/wiki/Field_trace">here</a> for some info and general properties of the
	* field trace.
	*
	* @note The field trace is a 'trace' in the linear algebra sense. We can view \f$ \mathbb{Q}(\zeta_9) \f$
	* as a vector space over the rationals, and an element \f$ a \f$ of this field gives a linear 
	* transformation \f$ x \longmapsto ax \f$ for each \f$ x \in \mathbb{Q}(\zeta_9) \f$. The trace of this
	* map is exactly the field trace here. The field trace is again guaranteed to be an integer.
	*******************************************************************************************************************/
	int fieldTrace() const;
	
	/** 
	* @brief Computes the product of all Galois conjugates except for the trivial Galois conjugate,
	* which is just the ring element itself.
	*
	* This function is used in @ref solveSystem (in exhaustive_search.cpp) to solve equations of the form
	* ax = b for \f$ x \f$ where \f$ a,b \in \mathbb{Z}[\zeta_9] \f$ if we let \f$ y \f$ denote the partial
	* norm of \f$ a \f$, then \f$ ay \f$ is an integer, and this fact makes it easier to determine whether
	* \f[
	* \frac{by}{ay}
	* \f]
	* is in \f$ \mathbb{Z}[\zeta_9,\frac{1}{3}] \f$. See @ref solveSystem for more details.
	*******************************************************************************************************************/
	ringZ9 partialFieldNorm() const;
	
	/**
	* @brief Performs field norm with respect to \f$ \mathbb{Q}(\zeta_9 + \zeta_9^8) \f$.
	* 
	* The field \f$ \mathbb{Q}(\zeta_9 + \zeta_9^8) \f$ is the unique real subfield of \f$ \mathbb{Q}(\zeta_9) \f$
	* and it has three Galois automorphisms (each of which are restrictions of those from \f$ \mathbb{Q}(\zeta_9) \f$).
	* Specifically, we multiply GaloisAut(1)*GaloisAut(2)*GaloisAut(5).
	* 
	* @warning This member function assumes that the ring element used is a REAL element. The function will return 
	* garbage if any nonreal element is passed.
	*******************************************************************************************************************/
	int tauFieldNorm() const;
	
	/**
	* @brief Counts the number of roots of unity used to express a ring element.
	* 
	* The weight is calculated by taking the sum of absolute values of each entry in @ref ringZ9::element.
	*
	* @note The weight of an arbitrary cyclotomic integer is not well-defined, but since we are 
	* representing the ring elements with a basis, this notion makes sense.
	*******************************************************************************************************************/
	int weight() const;
	
	/**
	* @brief Sums all coefficients.
	* 
	* Differs from @ref ringZ9::weight "weight" by summing over @ref ringZ9::element without using absolute values.
	* 
	* 
	* @note Much like the notion of weight, this function is not well-defined for arbitrary cyclotomic integers, but since we are 
	* representing the ring elements with a basis, this notion makes sense.
	*******************************************************************************************************************/
	int signedWeight() const;
	
	/**
	* @brief Computes the positive-definite quadratic form in Eq. (13) of [this paper](https://arxiv.org/abs/2311.08696).
	* 
	* If the coefficients of a ring element are given by \f$ a_i \f$ for \f$ i = 0,\ldots , 5 \f$, then the quadratic form
	* is given by
	* \f[
	* frac{1}{4}( 2a_0 - a_3)^2 + 3a_3^2 + (2a_1 - a_4)^2 + 3a_4^2 + (2a_2 - a_5)^2 + 3a_5^2
	* \f] \f[
	* = \sum_{i=0}^5 a_i^2 - a_0a_3 - a_1a_4 - a_2a_5.
	* \f]
	*******************************************************************************************************************/
	int quad() const;
	
	/**
	* @brief Viewing the ring element as a polynomial in the variable \f$ \zeta \f$, this function takes the derivative.
	* 
	* This function is used in @ref sdeChi to compute \f$ \mbox{sde}_\chi \f$ of a ring element.
	* 
	******************************************************************************************************************/
	ringZ9 formalDerivative() const;
	
	
	/**
	* @brief If \f$ \mbox{sde}_\chi \f$ of instance is less than 6, this computes it using Theorem 3.6 of the Kalra paper.
	* 
	* Theorem 3.6 in Kalra et. al. states that an element of this ring is divisible by \f$ \chi^k \f$ for \f$ k < 6 \f$ iff
	* the augmentation (see @ref ringZ9::signedWeight) of the \f$ (k-1)\f$-th formal derivative divided by \f$(k-1)!\f$
	* is divisible by 3. Our formal derivatives are calculated using @ref formalDerivative. 
	* 
	* The reason I created this is because Theorem 4.3 in the Kalra paper says that the sdes in terms of \f$ \chi \f$ in any
	* entries in a unit vector (and therefore, also a unitary matrix) will be the same. This means, when looking through candidates
	* for the diagonal and x_3 in @ref ESA, we can disregard any combination whose sde's disagree.
	* 
	* @note With our usage, we can assume this member function will only be called on ring elements whose \f$ \mbox{sde}_\chi \f$ is less
	* than 6 (since this is equivalent to the ring element being indivisible by 3). Thus, if this function is used on an element divisible
	* by 3, this function returns 6, even if the true sde is higher.  
	* 
	* @returns the sde in terms of \f$ \chi \f$ if less than 6, and 6 otherwise.
	******************************************************************************************************************/
	int sdeChi() const;
	/** @} */

	/* TESTS */
	/**
	* @defgroup ringZ9Tests ringZ9 Tests
	* @brief Member functions determining whether ring element satisfies nice properties.
	* @{
	*******************************************************************************************************************/
	/**
	* @brief Determines whether a ring element is zero.
	*
	* We check by determing whether the ring element can be expressed as a sum of "minimal
	* vanishing sums of roots of unity." For ninth roots of unity, these minimal vanishing sums
	* are precisely
	* \f[
	*		1 + \zeta_9^3 + \zeta_9^6 = 0
	* \f] \f[
	*		\zeta_9 + \zeta_9^4 + \zeta_9^7 = 0
	* \f] \f[
	*		\zeta_9^2 + \zeta_9^5 + \zeta_9^8 = 0
	* \f]
	*
	* @note When in reduced form, checking whether a ring element is zero simply amounts to 
	* checking that all entries of @ref ringZ9::element are zero, but in some cases, isZero is called before a
	* ring element passes through @ref ringZ9::reduce, so our algorithm is slightly more complicated.
	*******************************************************************************************************************/
	bool isZero() const;
	
	/**
	* @brief Checks whether ring element is real by seeing if ring is equal to @ref ringZ9::complexConj.
	*******************************************************************************************************************/
	bool isReal() const;
	/**
	* @brief Tests whether ring element is imaginary by seeing if it is equal to the negative of 
	* @ref complexConj.
	*******************************************************************************************************************/
	bool isImag() const;
	/**
	* @brief Determines whether ring element is integer by seeing if the only nonzero entry of @ref element
	* is the irst entry.
	*
	* @note Since we may assume the ring element has passed through @ref reduce
	*******************************************************************************************************************/
	bool isInt() const;
	
	/**
	* @brief Tests whether dividing the ring element by 'k' still lies in the ring.
	*
	* Testing this condition requires us to test whether each entry of @ref ringZ9::element is divisble
	* by k.
	*******************************************************************************************************************/
	bool isDivisibleByInt(int k) const;
	/** @} */
	
	/* PRINT */
	/** 
	* @brief Prints out element as \f$ \mathbb{Z} \f$-linear combination of ninth roots of unity.
	*******************************************************************************************************************/
	void print() const;
};


#endif
