#ifndef RING_H
#define RING_H

#include "cyclotomic_int9.h"

using namespace std;

/**
* @class ringZ9chi
* @brief Instances of the class represent elements of the ring \f$ \mathbb{Z}[\zeta_9, \frac{1}{3}] \f$ where
* \f$ \zeta_9 \f$ is the primitive ninth root of unity \f$ e^{2\pi/9} \f$.
*
* This class allows for basic arithmetic with elements of \f$ \mathbb{Z}[\zeta_9, \frac{1}{3}] \f$ and provides
* member functions that do some basic number-theoretic operations. Elements of this ring are comprised
* of a numerator which is a sum of ninth roots of unity  and a denominator which is 
* a nonnegative power of 3:
\f[
*  3^{-b}\cdot\sum_{i=0}^8 a_i \zeta_9^i
* \f]
* for \f$ a_i \in \mathbb{Z} \f$ and \f$ b \in \mathbb{Z}^{\geq 0} \f$.
* In this class, the varaible \f$ b \f$  is stored in @ref exp, and the \f$ a_i \f$'s are stored in 
* @ref ringZ9chi::element.
*
* @note The use of 'chi' in the class title is in reference to an alternative definition for this ring, where 
* we replace the generator 1/3 with \f$\chi := (1 - \zeta_9)^{-1} \f$. This replacement gives the same ring.
*
* @author Chris H.
* @date June 2025
**********************************************************************************************************************/
class ringZ9chi{
	
	private:
	/** 
	* @brief Array for the coefficients on each of the nine rou terms in the numerator.
	*
	* We may represent elements of our ring as a fraction where the numerator is an element of 
	* \f$ \mathbb{Z}[\zeta_9] \f$ which we encode with this array. As in @ref ringZ9::element,   
	* the \f$ i \f$-th entry of the array stores the integer coefficient on the \f$ \zeta_9^i \f$ term.
	*******************************************************************************************************************/
	int element[9];
	
	/**
	* @brief This variable stores the power of three in the denominator
	*
	* We may represent elements of this ring as a fraction where the denominator is a nonnegative
	* integer power of three, which is stored in this variable. I could have made the variable not 
	* unsigned but when combined with @ref ringZ9chi::cancel(), testing for membership in 
	* \f$ \mathbb{Z}[\zeta_9] \f$ now amounts to testing whether this variable is zero.
	*******************************************************************************************************************/
	int exp;
	
	public:
	/* CONSTRUCTORS */
	/** 
	* @defgroup ringZ9chiConstructors ringZ9chi Constructors
	* @brief Various constructors for an element
	* @{
	*******************************************************************************************************************/
	ringZ9chi(int arr[9], int denom_exp); ///< Defines element explicitly by copying 'arr' to @ref ringZ9chi::element and 'denom_exp' to @ref exp.
	ringZ9chi(ringZ9 numer, int denom_exp); ///< Defines element explicitly by using a @ref ringZ9 for the numerator and copies 'denom_exp' to @ref exp.
	ringZ9chi(); ///< Constructor with no arguments returns zero element.
	/** @} */

	/* INFO */
	/** 
	* @defgroup ringZ9chiInfo ringZ9chi Info
	* @brief Member functions returning specific information on ring element.
	* @{
	*******************************************************************************************************************/
	int getTerm(int i) const; ///< Returns \f$ i \f$-th entry of @ref ringZ9chi::element giving the coefficient of the \f$ \zeta_9^i \f$ term in the numerator.
	int getExp() const; ///< Returns the exponent on the three in the denominator.
	ringZ9 getNumerator() const; ///< Returns the numerator as an element of @ref ringZ9.
	array<int, 6> getStdArray() const; ///< Returns @ref ringZ9chi::element as a std::array.
	/** @} */
	
	/* MUTATORS */
	/**
	* @defgroup ringZ9chiMutators ringZ9chi Mutators
	* @brief Member functions which have authority to modify the ring element itself.
	* @{
	*******************************************************************************************************************/
	/**
	* @brief Called by certain constructors and after certain operations to simplify ring element into a canonical form.
	*
	* First tests whether ring element is zero element. If not, then @ref ringZ9chi::reduce and @ref cancel are 
	* called. This function puts the ring element in a canonical form so that testing equality between two
	* normalized elements amounts to checking whether @ref ringZ9chi::element and @ref exp are 
	* identical for both.
	*
	* @warning Most member functions for this class are designed assuming that any given ring 
	* element is normalized. Any constructor or member function which has a chance of producing 
	* a non-normalized ring element must call this function (or @ref ringZ9chi::reduce or @ref cancel
	* depnding on context) before terminating.
	*******************************************************************************************************************/
	void normalize();
	
	/**
	* @brief Writes numerator in terms of the canonical \f$ \mathbb{Z} \f$-basis for \f$ \mathbb{Z}[\zeta_9] \f$.
	*
	* See @ref ringZ9::reduce for specific details.
	*******************************************************************************************************************/
	void reduce();
	
	/**
	* @brief Simplifies ring element by cancelling largest power of three shared between numerator 
	* and denominator.
	*
	* An element of \f$ \mathbb{Z} [\zeta_9] \f$ in reduced form is divisible by 3 in said ring iff each
	* coefficient is divisible by 3. So, the algorithm divides each coefficient by three and decrements 
	* the exponent in the denominator untile either the exponent is zero or there is at least one term in
	* the numerator whose coefficient is indivisibly by 3.
	*
	* @warning When a new instance of this class is defined, this function must be called AFTER 
	* @ref ringZ9chi::reduce is called. Otherwise, an unreduced numerator may still be divisible by 3
	* even if there are terms which are indivisible by 3.
	*******************************************************************************************************************/
	void cancel();
	
	void scalar_mult(int scalar); ///< Multiplies each term of the numerator by 'scalar' and calls @ref ringZ9chi::reduce.
	/** @} */
	
	/* OPERATIONS */
	/**
	* @defgroup ringZ9chiBasicOperations ringZ9chi Basic Operations
	* @brief Operations of addition, subtraction, multiplication, and assignment defined in the usual way.
	* @{
	*******************************************************************************************************************/
	ringZ9chi operator+(const ringZ9chi& right) const; ///< Ring addition
	ringZ9chi operator-(const ringZ9chi& right) const; ///< Ring subtraction
	ringZ9chi operator*(const ringZ9chi& right) const; ///< Multiplies ring elements and then passes product through @ref ringZ9chi::reduce
	ringZ9chi operator=(const ringZ9chi& right); ///< Ring assignment
	/** @} */
	
	/**
	* @defgroup ringZ9chiEquality ringZ9chi Equality
	* Since we may assume both sides have passed through @ref ringZ9chi::reduce @ref cancel, it 
	* suffices to check whether both @ref ringZ9chi::element arrays and @ref exp are identical to 
	* verify equality.
	* @{
	*******************************************************************************************************************/
	bool operator==(const ringZ9chi& right) const; ///< Ring equality
	bool operator!=(const ringZ9chi& right) const; ///< Ring ineqality
	/** @} */
	
	/* TYPE CONVERSIONS */
	/**
	* @defgroup ringZ9chiTypeConversions ringZ9chi Type Conversions
	* @{ 
	*******************************************************************************************************************/
	double real_part() const; ///< Real part 
	double imag_part() const; ///< Imaginary part
	double abs_val() const; ///< Absolute value
	double abs_val_sq() const; ///< Absolute value squared
	complex<double> toComplexDouble() const; ///< Converts ring element to std::complex<double> 
	/** @} */
	
	/* NUMBER THEORETIC */
	/**
	* @defgroup ringZ9chiNumberTheoretic ringZ9chi Number-theoretic
	* @brief Some basic number-theretic routines.
	* @{
	*******************************************************************************************************************/
	/** 
	* @brief Performs one of six distinct Galois automorphisms on ring element.
	*
	* See @ref ringZ9::GaloisAut for more information about Galois automorphisms. Since Galois 
	* automorphisms act trivially on rational numbers, we only need to apply the Galois automorphism
	* on the numerator. Afterwards, the result is reduced.
	*******************************************************************************************************************/
	ringZ9chi GaloisAut(int k) const;
	
	/**
	* @brief Computes complex conjugate by returning GaloisAut(-1).
	*******************************************************************************************************************/
	ringZ9chi complexConj() const;
	
	/**
	* @brief Computes field norm
	*
	* See @ref ringZ9::fieldNorm for specific details.
	*******************************************************************************************************************/
	double fieldNorm() const;
	
	/**
	* @brief Computes field trace
	*
	* See @ref ringZ9::fieldTrace for specific details.
	*******************************************************************************************************************/
	double fieldTrace() const;
	/** @} */
	
	/* TESTS */
	/**
	* @defgroup ringZ9chiTests ringZ9chi Tests
	* @brief Member functions determining whether ring element satisfies nice properties.
	* @{
	*******************************************************************************************************************/
	bool isZero() const; ///< Tests whether ring element is zero by seeing if numerator is zero. See @ref ringZ9::isZero for specific details.
	bool isReal() const; ///< Tests whether ring element is real by seeing if ring element equals its complex conjugate
	bool isImag() const; ///< Tests whether ring element is imaginary by seeing if ring element equals the negative of its complex conjugate
	bool isRational() const; ///< A ring element will be rational iff the first entry of @ref ringZ9chi::element is the only nonzero entry.
	bool isInt() const; ///< Tests if ring element is integer by seeing whether @ref isRational is true and @ref exp = 0.
	bool isCycInt() const; ///< Tests if ring element lies in \f$ \mathbb{Z}[\zeta_9] \f$ by seeing if @ref exp = 0.
	
	/** @} */
	
	/* PRINT */
	/** 
	* @brief Prints out element as \f$ \mathbb{Z} \f$-linear combination of ninth roots of unity along with
	* the appropriate power of 1/3 out front.
	*******************************************************************************************************************/
	void print() const;
};


#endif
