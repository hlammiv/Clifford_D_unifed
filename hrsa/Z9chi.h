#ifndef RING_H
#define RING_H

#include "cyclotomic_int9.h"

using namespace std;

/**
* @class ringZ9chiBase
* @brief Templated cyclotomic-with-1/3 ring class for Z[zeta_9, 1/3].
*
* This is a templated rewrite (2026-05) of the original int-only ringZ9chi
* class.  Instantiate with T=int for the HRSA hot path and
* T=boost::multiprecision::cpp_int for decompose() on SK output.
*
* The default typedef `using ringZ9chi = ringZ9chiBase<int>` preserves all
* existing call sites.  See Z9chi.h.orig.bak for the pre-templated reference.
**********************************************************************************************************************/
template<typename T>
class ringZ9chiBase{

	private:
	T element[9];
	int exp;

	// Integer power-of-3 helper — avoids double truncation from std::pow.
	static T int_pow3(int n) {
		T result = T(1);
		for (int i = 0; i < n; ++i) result *= T(3);
		return result;
	}

	template<typename U> friend class ringZ9chiBase;

	public:
	/* CONSTRUCTORS */
	ringZ9chiBase(const T arr[9], int denom_exp){
		exp = denom_exp;
		for(int i = 0; i < 9; ++i) {
			element[i] = arr[i];
		}
		normalize();
	}

	// Overload for literal int[9] arrays at call sites (decompose.cpp etc.),
	// only enabled when T != int to avoid ambiguity.
	template<typename U>
	ringZ9chiBase(const U arr[9], int denom_exp,
	              typename std::enable_if<!std::is_same<U, T>::value, int>::type = 0){
		exp = denom_exp;
		for(int i = 0; i < 9; ++i){
			element[i] = T(arr[i]);
		}
		normalize();
	}

	ringZ9chiBase(const ringZ9Base<T>& numer, int denom_exp){
		exp = denom_exp;
		for(int i = 0; i < 6; i++){
			element[i] = numer.getTerm(i);
		}
		for(int i = 6; i < 9; i++){
			element[i] = T(0);
		}
		cancel();
	}

	ringZ9chiBase(){
		exp = 0;
		for(int i = 0; i < 9; i++){
			element[i] = T(0);
		}
	}

	// Cross-template copy.
	template<typename U>
	ringZ9chiBase(const ringZ9chiBase<U>& other,
	              typename std::enable_if<!std::is_same<U, T>::value, int>::type = 0){
		exp = other.getExp();
		for(int i = 0; i < 6; ++i){
			element[i] = T(other.getTerm(i));
		}
		for(int i = 6; i < 9; ++i){
			element[i] = T(0);
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

	int getExp() const {
		return exp;
	}

	ringZ9Base<T> getNumerator() const{
		T copy[9];
		for(int i = 0; i < 9; ++i) copy[i] = T(0);
		for(int i = 0; i < 6; i++){
			copy[i] = element[i];
		}
		return ringZ9Base<T>(copy);
	}

	array<T, 6> getStdArray() const{
		array<T, 6> arr;
		for(int i = 0; i < 6; i++){
			arr[i] = element[i];
		}
		return arr;
	}

	/* MUTATORS */
	void normalize(){
		if(isZero()){
			exp = 0;
			for(int i = 0; i < 9; i++){
				element[i] = T(0);
			}
			return;
		}
		reduce();
		cancel();
	}

	void reduce() {
		T e6 = element[6];
		T e7 = element[7];
		T e8 = element[8];

		element[0] -= e6;
		element[3] -= e6;

		element[1] -= e7;
		element[4] -= e7;

		element[2] -= e8;
		element[5] -= e8;

		element[6] = T(0);
		element[7] = T(0);
		element[8] = T(0);
	}

	void cancel(){
		bool unfinished = true;
		while(unfinished && exp > 0){
			for(int i = 0; i < 6; i++){
				if((element[i] % T(3)) != T(0)){
					unfinished = false;
					break;
				}
			}
			if(unfinished){
				exp--;
				for(int i = 0; i < 6; i++){
					element[i] = element[i]/T(3);
				}
			}
		}
	}

	void scalar_mult(const T& scalar){
		for(int i = 0; i < 9; i++){
			element[i] = scalar*element[i];
		}
		cancel();
	}

	/* OPERATIONS */
	ringZ9chiBase operator+(const ringZ9chiBase& right) const{
		ringZ9chiBase sum;
		if(exp < right.exp){
			sum.exp = right.exp;
			T scale = int_pow3(right.exp - exp);
			for(int i = 0; i < 6; i++){
				sum.element[i] = scale*element[i] + right.element[i];
			}
		} else if(exp > right.exp){
			sum.exp = exp;
			T scale = int_pow3(exp - right.exp);
			for(int i = 0; i < 6; i++){
				sum.element[i] = element[i] + scale*right.element[i];
			}
		} else {
			sum.exp = exp;
			for(int i = 0; i < 6; i++){
				sum.element[i] = element[i] + right.element[i];
			}
		}
		sum.cancel();
		return sum;
	}

	ringZ9chiBase operator-(const ringZ9chiBase& right) const{
		ringZ9chiBase diff;
		if(exp < right.exp){
			diff.exp = right.exp;
			T scale = int_pow3(right.exp - exp);
			for(int i = 0; i < 6; i++){
				diff.element[i] = scale*element[i] - right.element[i];
			}
		} else if(exp > right.exp){
			diff.exp = exp;
			T scale = int_pow3(exp - right.exp);
			for(int i = 0; i < 6; i++){
				diff.element[i] = element[i] - scale*right.element[i];
			}
		} else {
			diff.exp = exp;
			for(int i = 0; i < 6; i++){
				diff.element[i] = element[i] - right.element[i];
			}
		}
		diff.cancel();
		return diff;
	}

	ringZ9chiBase operator*(const ringZ9chiBase& right) const{
		ringZ9chiBase prod;
		prod.exp = exp + right.exp;
		for(int i = 0; i <9; i++){
			for(int j = 0; j < 9; j++)
				prod.element[(i + j)%9] = prod.element[(i + j)%9] + element[i]*right.element[j];
		}
		prod.normalize();
		return prod;
	}

	ringZ9chiBase& operator=(const ringZ9chiBase& right) {
		if (this != &right) {
			for (int i = 0; i < 9; ++i){
				this->element[i] = right.element[i];
			}
			this->exp = right.exp;
		}
		return *this;
	}

	bool operator==(const ringZ9chiBase& right) const{
		if (exp != right.exp) return false;
		for (int i = 0; i < 6; ++i)
			if (element[i] != right.element[i]) return false;
		return true;
	}

	bool operator!=(const ringZ9chiBase& right) const{
		if (exp != right.exp) return true;
		for (int i = 0; i < 6; ++i)
			if (element[i] != right.element[i]) return true;
		return false;
	}

	/* TYPE CONVERSIONS */
	double real_part() const{
		static const double cos_vals[6] = {
			1.0,
			0.766044443118978,
			0.17364817766693041,
			-0.5,
			-0.9396926207859083,
			-0.9396926207859084,
		};
		double scale = 1.0;
		for (int i = 0; i < exp; ++i) scale /= 3.0;

		double sum = to_double(element[0]) * cos_vals[0];
		for (int i = 1; i < 6; ++i)
			sum += cos_vals[i] * to_double(element[i]);
		return sum * scale;
	}

	double imag_part() const{
		static const double sin_vals[6] = {
			0.0,
			0.6427876096865393,
			0.984807753012208,
			0.8660254037844387,
			0.3420201433256689,
			-0.34202014332566866,
		};
		double scale = 1.0;
		for (int i = 0; i < exp; ++i) scale /= 3.0;

		double sum = 0.0;
		for (int i = 1; i < 6; ++i)
			sum += sin_vals[i] * to_double(element[i]);
		return sum * scale;
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

	/* NUMBER THEORETIC */
	ringZ9chiBase complexConj() const{
		T swap[9] = {T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0)};
		for (int i = 0; i < 6; i++){
			swap[(8*i)%9] = element[i];
		}
		return ringZ9chiBase(swap,exp);
	}

	ringZ9chiBase GaloisAut(int k) const{
		k = k%9;
		T swap[9] = {T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0)};
		for (int i = 0; i < 6; i++){
			swap[(((k*i % 9) + 9) % 9)] = swap[(((k*i % 9) + 9) % 9)] + element[i];
		}
		return ringZ9chiBase(swap,exp);
	}

	double fieldNorm() const{
		ringZ9chiBase prod = *this;
		for(int i = 2; i < 9; i++){
			if(gcd(i,9) == 1){
				prod = prod*GaloisAut(i);
			}
		}
		prod.reduce();
		return to_double(prod.getTerm(0));
	}

	double fieldTrace() const{
		ringZ9chiBase sum = *this;
		for(int i = 1; i < 9; i++){
			if(gcd(i,9) == 1){
				sum = sum+GaloisAut(i);
			}
		}
		return to_double(sum.getTerm(0));
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
		ringZ9chiBase test = *this - this->complexConj();
		return test.isZero();
	}

	bool isImag() const{
		ringZ9chiBase test = *this + this->complexConj();
		return test.isZero();
	}

	bool isRational() const{
		for(int i = 1; i < 6; i++){
			if(element[i] != T(0)){
				return false;
			}
		}
		return true;
	}

	bool isInt() const{
		if(isRational() && (element[0] % int_pow3(exp)) == T(0)){
			return true;
		}
		return false;
	}

	bool isCycInt() const{
		return exp == 0;
	}

	/* PRINT */
	void print() const{
		if(isZero()){
			cout << "(0)";
			return;
		}
		if(exp != 0){
			cout << "(1/3^" << exp << ")";
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

	private:
	static double to_double(int v) { return (double)v; }
	static double to_double(long long v) { return (double)v; }
	static double to_double(const boost::multiprecision::cpp_int& v) {
		return v.template convert_to<double>();
	}
};


// ---- public typedefs --------------------------------------------------------
using ringZ9chi    = ringZ9chiBase<int>;
using ringZ9chiBig = ringZ9chiBase<boost::multiprecision::cpp_int>;


#endif
