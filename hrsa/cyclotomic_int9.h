#ifndef RINGZ9_H
#define RINGZ9_H

#include<iostream>
#include<cmath>
#include<array>
#include<numeric>
#include<complex>
#include<cstdlib>
#include<boost/multiprecision/cpp_int.hpp>

# ifndef M_PI
# define M_PI           3.14159265358979323846
# endif

using namespace std;

/**
* @class ringZ9Base
* @brief Templated cyclotomic ring class for Z[zeta_9]; coefficients stored as type T.
*
* This is a templated rewrite (2026-05) of the original int-only ringZ9 class.
* Instantiate with T=int for the int32 hot path (HRSA's enumeration loops where
* the per-element coefficients always fit in int32) and with
* T=boost::multiprecision::cpp_int for the big-int post-processing path
* (decompose() called on SK-assembled exact unitaries whose coefficients can be
* O(3^f) for f≥40, well outside int64 range).
*
* The default typedef `using ringZ9 = ringZ9Base<int>` at the bottom of this
* header preserves all existing call sites: HRSA_test, householder_search,
* bidir_reify, rf_features, hrsa_v_inspect, f0_cache_builder, cliff_index_check,
* and e0_net_dump all continue to compile and run unchanged.
*
* For documentation of the semantics of each member, see the original int-only
* version (preserved in cyclotomic_int9.h.orig.bak).
**********************************************************************************************************************/
template<typename T>
class ringZ9Base{

	private:
	T element[9];

	public:

	/* CONSTRUCTORS */
	ringZ9Base(const T arr[9]){
		for(int i = 0; i < 9; ++i) {
			element[i] = arr[i];
		}
		reduce();
	}

	// Convenience overload: still construct from a literal int[9] when T != int.
	// Needed by call sites in decompose.cpp that build `int arr[9] = {0,...}`
	// then construct a ringZ9.  When T==int, this overload collapses to the
	// above (one of them, picked by overload resolution).
	template<typename U>
	ringZ9Base(const U arr[9],
	           typename std::enable_if<!std::is_same<U, T>::value, int>::type = 0){
		for(int i = 0; i < 9; ++i){
			element[i] = T(arr[i]);
		}
		reduce();
	}

	ringZ9Base(int n){
		element[0] = T(n);
		for(int i = 1; i < 9; i++){
			element[i] = T(0);
		}
	}

	ringZ9Base(int coeff, int term){
		for(int i = 0; i < 9; i++){
			if(i == term%9){
				element[i] = T(coeff);
			} else {
				element[i] = T(0);
			}
		}
		reduce();
	}

	ringZ9Base(){
		for(int i = 0; i < 9; i++){
			element[i] = T(0);
		}
	}

	// Cross-template copy: allow `ringZ9Big a = ringZ9_int_value;` style conversions.
	template<typename U>
	ringZ9Base(const ringZ9Base<U>& other,
	           typename std::enable_if<!std::is_same<U, T>::value, int>::type = 0){
		for(int i = 0; i < 9; ++i){
			element[i] = T(other.getTerm(i));
		}
	}

	/* INFO */
	T getTerm(int i) const {
		if(i > 8 || i < 0){
			return T(0);
		} else {
			return element[i];
		}
	}

	array<T, 6> getStdArray() const{
		array<T, 6> arr;
		for(int i = 0; i < 6; i++){
			arr[i] = element[i];
		}
		return arr;
	}

	/* MUTATORS */
	void reduce(){
		if(isZero()){
			for(int i = 0; i < 9; i++){
				element[i] = T(0);
			}
			return;
		}

		for(int i = 6; i < 9; i++){
			for(int j = 1; j < 3; j++){
				element[(i + 3*j)%9] = element[(i + 3*j)%9] - element[i];
			}
		}
		element[6] = T(0);
		element[7] = T(0);
		element[8] = T(0);
	}

	void scalar_mult(const T& scalar){
		for(int i = 0; i < 9; i++){
			element[i] = scalar*element[i];
		}
	}

	void scalar_div(const T& scalar){
		if(scalar == T(0)){
			cout << "Zero passed as scalar in ringZ9Base::scalar_div" << endl;
		}else {
			for(int i = 0; i < 9; i++){
				element[i] = element[i]/scalar;
			}
		}
	}

	/* OPERATIONS */
	ringZ9Base operator+(const ringZ9Base& right) const{
		ringZ9Base sum;
		for(int i = 0; i < 6; i++){
			sum.element[i] = element[i] + right.element[i];
		}
		return sum;
	}

	ringZ9Base operator-(const ringZ9Base& right) const{
		ringZ9Base diff;
		for(int i = 0; i < 6; i++){
			diff.element[i] = element[i] - right.element[i];
		}
		return diff;
	}

	ringZ9Base operator*(const ringZ9Base& right) const{
		ringZ9Base prod;
		for(int i = 0; i <6; i++){
			for(int j = 0; j < 6; j++){
				prod.element[(i + j)%9] = prod.element[(i + j)%9] + element[i]*right.element[j];
			}
		}
		prod.reduce();
		return prod;
	}

	ringZ9Base operator*(const T& right_scalar) const{
		ringZ9Base prod;
		for(int i = 0; i < 6; i++){
			prod.element[i] = element[i]*right_scalar;
		}
		return prod;
	}

	ringZ9Base operator/(const T& right_scalar) const{
		ringZ9Base prod;
		if(right_scalar == T(0)){
			cout << "Zero passed as scalar in ringZ9Base::operator/" << endl;
			return prod;
		}
		for(int i = 0; i < 6; i++){
			prod.element[i] = element[i]/right_scalar;
		}
		return prod;
	}

	ringZ9Base& operator=(const ringZ9Base& right) {
		if (this != &right) {
			for (int i = 0; i < 9; ++i)
				this->element[i] = right.element[i];
		}
		return *this;
	}

	bool operator==(const ringZ9Base& right) const{
		for (int i = 0; i < 6; ++i)
			if (element[i] != right.element[i]) return false;
		return true;
	}

	bool operator!=(const ringZ9Base& right) const{
		for (int i = 0; i < 6; ++i)
			if (element[i] != right.element[i]) return true;
		return false;
	}

	/* TYPE CONVERSIONS */
	double real_part() const {
		static const double cos_vals[4] = {
			0.766044443118978,
			0.17364817766693041,
			-0.5,
			-0.9396926207859083,
		};
		double sum = to_double(element[0]);
		sum += cos_vals[0] * to_double(element[1]);
		sum += cos_vals[1] * to_double(element[2]);
		sum += cos_vals[2] * to_double(element[3]);
		sum += cos_vals[3] * (to_double(element[4])+to_double(element[5]));
		return sum;
	}

	double imag_part() const {
		static const double sin_vals[4] = {
			0.6427876096865393,
			0.984807753012208,
			0.8660254037844387,
			0.3420201433256689,
		};
		double sum = 0.0;
		sum += sin_vals[0] * to_double(element[1]);
		sum += sin_vals[1] * to_double(element[2]);
		sum += sin_vals[2] * to_double(element[3]);
		sum += sin_vals[3] * (to_double(element[4])-to_double(element[5]));
		return sum;
	}

	double complexArg() const{
		return atan2(imag_part(), real_part());
	}

	double abs_val() const{
		return sqrt(real_part()*real_part() + imag_part()*imag_part());
	}

	double abs_val_sq() const{
		return real_part()*real_part() + imag_part()*imag_part();
	}

	complex<double> toComplexDouble() const{
		complex<double> z(real_part(),imag_part());
		return z;
	}

	/* NUMBER-THEORETIC */
	ringZ9Base complexConj() const{
		T swap[9] = {T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0)};
		for (int i = 0; i < 6; i++){
			swap[(8*i)%9] = element[i];
		}
		return ringZ9Base(swap);
	}

	ringZ9Base GaloisAut(int k) const{
		k = k%9;
		T swap[9] = {T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0)};
		for (int i = 0; i < 6; i++){
			swap[(((k*i % 9) + 9) % 9)] = swap[(((k*i % 9) + 9) % 9)] + element[i];
		}
		return ringZ9Base(swap);
	}

	T fieldNorm() const{
		ringZ9Base prod(1);
		for(int i = 1; i < 9; i++){
			if(gcd(i,9) == 1){
				prod = prod*GaloisAut(i);
			}
		}
		prod.reduce();
		return prod.getTerm(0);
	}

	T fieldTrace() const{
		ringZ9Base sum = GaloisAut(1);
		for(int i = 2; i < 9; i++){
			if(gcd(i,9) == 1){
				sum = sum+GaloisAut(i);
			}
		}
		return sum.getTerm(0);
	}

	ringZ9Base partialFieldNorm() const{
		ringZ9Base prod = GaloisAut(2);
		for(int i = 4; i < 9; i++){
			if(gcd(i,9) == 1){
				prod = prod*GaloisAut(i);
			}
		}
		return prod;
	}

	T tauFieldNorm() const{
		ringZ9Base prod = GaloisAut(1)*GaloisAut(2)*GaloisAut(5);
		return prod.getTerm(0);
	}

	T weight() const{
		T sum = T(0);
		for(int i = 0; i < 6; i++){
			sum = sum + ring_abs(element[i]);
		}
		return sum;
	}

	T signedWeight() const{
		T sum = T(0);
		for(int i = 0; i < 6; i++){
			sum = sum + element[i];
		}
		return sum;
	}

	T quad() const{
		T sum = T(0);
		for(int i = 0; i < 6; i++){
			sum = sum + element[i]*element[i];
		}
		return sum - element[0]*element[3] - element[1]*element[4] - element[2]*element[5];
	}

	ringZ9Base formalDerivative() const{
		T new_element[9];
		for(int i = 0; i < 9; ++i) new_element[i] = T(0);
		for(int i = 0; i < 6; i++){
			new_element[i] = T(i+1)*element[i+1];
		}
		return ringZ9Base(new_element);
	}

	int sdeChi() const{
		// signedWeight() may be negative; mod-3 normalization required.
		if (mod3_nonzero(signedWeight())) return 0;

		ringZ9Base test = formalDerivative();

		for(int i = 1; i < 6; i++){
			T sw = test.signedWeight() / T(i);
			if (mod3_nonzero(sw)) return i;
			test = (test.formalDerivative())/T(i);
		}
		return 6;
	}

	/* TESTS */
	bool isZero() const{
		for(int i = 0; i < 3; i++){
			for(int j = 1; j < 3; j++){
				if(element[i + 3*j] !=element[i]){
					return false;
				}
			}
		}
		return true;
	}

	bool isReal() const{
		ringZ9Base test = *this - this->complexConj();
		return test.isZero();
	}

	bool isImag() const{
		ringZ9Base test = *this + this->complexConj();
		return test.isZero();
	}

	bool isInt() const{
		for(int i = 1; i < 6; i++){
			if(element[i] != T(0)){
				return false;
			}
		}
		return true;
	}

	bool isDivisibleByInt(int k) const{
		for(int i = 0; i < 6; i++){
			if(mod_int(element[i], k) != 0){
				return false;
			}
		}
		return true;
	}

	/* PRINT */
	void print() const{
		if(isZero()){
			cout << "(0)" << endl;
			return;
		}
		int i = 0;
		int j = 8;
		while(element[i] == T(0)){ i++;}
		while(element[j] == T(0)){	j--;}
		if(i == j){
			cout << "(" << element[i] << "z^" << i << ")";
			return;
		}
		cout << "(";
		for(int k = i; k < j; k++){
			if(element[k] != T(0)){
				cout << element[k] << "z^" << k << " + ";
			}
		}
		cout << element[j] << "z^" << j << ")";
	}

	// ---- type-specific helpers (free static functions, picked via ADL) ----
private:
	// double conversion: built-in for int, multiprecision::cpp_int needs explicit cast.
	static double to_double(int v) { return (double)v; }
	static double to_double(long long v) { return (double)v; }
	static double to_double(const boost::multiprecision::cpp_int& v) { return v.template convert_to<double>(); }

	// abs for whatever T is.  For cpp_int there's a free `abs()` ADL hook.
	static T ring_abs(const T& v) {
		using std::abs;
		using boost::multiprecision::abs;
		return abs(v);
	}

	// mod3_nonzero: returns true iff v % 3 != 0.  For cpp_int we use a free
	// modulo operator (which is well-defined for non-negative; for negatives
	// we take abs first so sign doesn't matter for "is divisible by 3").
	static bool mod3_nonzero(const T& v) {
		T r = v % T(3);
		return r != T(0);
	}

	// integer modulo by a built-in int (positive divisor).  Returns 0 iff
	// divisible.  For negative T this is the C++ truncation-toward-zero
	// remainder; we only use the result against 0 in isDivisibleByInt so the
	// sign convention doesn't affect correctness.
	static int mod_int(const T& v, int k) {
		T r = v % T(k);
		return r == T(0) ? 0 : 1;
	}

	template<typename U> friend class ringZ9Base;
};


// ---- public typedefs --------------------------------------------------------
//
// `ringZ9`     : the int-only ring (default).  All HRSA hot-path code uses this.
// `ringZ9Big`  : the cpp_int ring, for decompose() on SK-assembled unitaries
//                where coefficients can be O(3^f) for f ≥ 40.
using ringZ9    = ringZ9Base<int>;
using ringZ9Big = ringZ9Base<boost::multiprecision::cpp_int>;


#endif
