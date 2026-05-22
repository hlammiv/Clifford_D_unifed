#include "cyclotomic_int9.h"

	/* *********************** */
	/* CONSTRUCTORS  */
	/* *********************** */
	ringZ9::ringZ9(int arr[9]){
		
		for(int i = 0; i < 9; ++i) {
			element[i] = arr[i];
		}
		
		reduce();
	}
	

	ringZ9::ringZ9(int n){
		element[0] = n;
		
		for(int i = 1; i < 9; i++){
			element[i] = 0;
		}
	}
	

	ringZ9::ringZ9(int coeff, int term){
		
		for(int i = 0; i < 9; i++){
			if(i == term%9){
				element[i] =coeff;
			} else {
				element[i] = 0;
			}
		}
		
		reduce();
	}
	
	ringZ9::ringZ9(){

		for(int i = 0; i < 9; i++){
				element[i] = 0;
		}
	}
	
	/* ******* */
	/* INFO */
	/* ******* */

	int ringZ9::getTerm(int i) const {
		if(i > 8 || i < 0){
			return 0;
		} else {
			return element[i];
		}	
	}
	
	

	array<int, 6> ringZ9::getStdArray() const{
		
		array<int, 6> arr;
		
		for(int i = 0; i < 6; i++){
			arr[i] = element[i];
		}
		return arr;
	}


	/* *************** */
	/* MUTATORS */
	/* *************** */

	// Reduce to canonical form (slots 6..8 zero) using ζ_9^{6+k} = -ζ_9^k - ζ_9^{3+k}.
	//
	// Fix E.2 (2026-05): early-exit for already-canonical inputs, drop the
	// isZero() shortcut, unroll the relation, eliminate the modulo.
	//
	// Why dropping the isZero() shortcut is safe: the original code zeroed
	// all 9 slots when isZero() returned true (catching cyclotomic-zero
	// inputs like [c,c,c,c,c,c,c,c,c]).  The unrolled three subtractions
	// below already collapse such inputs to all-zeros — so the shortcut
	// was only saving the work, not preserving any semantic.
	void ringZ9::reduce(){
		if (element[6] == 0 && element[7] == 0 && element[8] == 0) return;

		// ζ_9^6 = -ζ_9^3 - 1   →   element[0] and element[3] absorb -element[6]
		element[0] -= element[6];
		element[3] -= element[6];
		element[6] = 0;

		// ζ_9^7 = -ζ_9^4 - ζ_9
		element[1] -= element[7];
		element[4] -= element[7];
		element[7] = 0;

		// ζ_9^8 = -ζ_9^5 - ζ_9^2
		element[2] -= element[8];
		element[5] -= element[8];
		element[8] = 0;
	}
	
	
	void ringZ9::scalar_mult(int scalar){
		for(int i = 0; i < 9; i++){
			element[i] = scalar*element[i];
		}
	}
	
	void ringZ9::scalar_div(int scalar){
		if(scalar == 0){
			cout << "Zero passed as scalar in ringZ9::scalar_div(int scalar)" << endl;
		}else {		
			for(int i = 0; i < 9; i++){
				element[i] = element[i]/scalar;
			}
		}
	}
	
	
	
	/* ***************** */
	/* OPERATIONS */
	/* ***************** */
	ringZ9 ringZ9::operator+(const ringZ9& right) const{
		ringZ9 sum;
		
		for(int i = 0; i < 6; i++){
			sum.element[i] = element[i] + right.element[i];
		}

		return sum;
	}
	
	
	ringZ9 ringZ9::operator-(const ringZ9& right) const{
		ringZ9 diff;
		
		for(int i = 0; i < 6; i++){
			diff.element[i] = element[i] - right.element[i];
		}

		return diff;
	}
	
	// Fused multiply-and-reduce (Fix E.1, 2026-05).
	//
	// The previous implementation wrote partial products into a 9-slot
	// accumulator (using `(i+j)%9` per inner step), then ran reduce() to
	// fold slots 6-8 back into slots 0-5 via the relation ζ_9^6 = -ζ_9^3 - 1.
	// We do both in one pass.
	//
	// Implementation: split the (i,j) ∈ [0,5]^2 grid by their sum s = i+j
	// (s ∈ [0,10]) into three blocks the compiler can keep in registers:
	//   block A — pairs with s ∈ [0,5]:  prod[s]     += a[i]*b[j].
	//   block B — pairs with s ∈ [6,8]:  prod[s-6]   -= a[i]*b[j];
	//                                    prod[s-3]   -= ...   (ζ^{s} = -ζ^{s-3} - ζ^{s-6}).
	//   block C — pairs with s ∈ [9,10]: prod[s-9]   += ...   (ζ^9 = 1, ζ^10 = ζ).
	//
	// Static loops (constexpr-bounded) let the optimizer fully unroll,
	// avoiding the runtime modulo/switch entirely.  Slots 6-8 of `prod`
	// are never written, so no reduce() is needed.  Behavior is byte-for-
	// byte identical to the previous code (verified by ringZ9_unit_test).
	ringZ9 ringZ9::operator*(const ringZ9& right) const{
		ringZ9 prod;
		const int* a = element;
		const int* b = right.element;
		int* p = prod.element;

		// Block A: i + j ≤ 5.  Loop bounds keep s ∈ [0, 5].
		for(int i = 0; i < 6; ++i){
			const int ai = a[i];
			const int jmax = 5 - i;       // j ∈ [0, 5 - i]
			for(int j = 0; j <= jmax; ++j){
				p[i + j] += ai * b[j];
			}
		}

		// Block B: i + j ∈ {6, 7, 8}.  ζ^s = -ζ^{s-3} - ζ^{s-6}.
		// (i, j) pairs with i + j = s and i, j ∈ [0,5]:
		//   s=6: (1,5),(2,4),(3,3),(4,2),(5,1)         — i ∈ [1,5]
		//   s=7: (2,5),(3,4),(4,3),(5,2)               — i ∈ [2,5]
		//   s=8: (3,5),(4,4),(5,3)                     — i ∈ [3,5]
		// Equivalently: i ∈ [1,5], j ∈ [max(6-i, 0), 5], s = i+j ∈ [6, i+5].
		for(int i = 1; i < 6; ++i){
			const int ai = a[i];
			const int jmin = (6 - i > 0) ? 6 - i : 0;
			for(int j = jmin; j < 6; ++j){
				const int s = i + j;       // ∈ [6, 10]; we want only [6,8] here
				if (s > 8) break;
				const int v = ai * b[j];
				p[s - 6] -= v;
				p[s - 3] -= v;
			}
		}

		// Block C: i + j ∈ {9, 10}.  Only (4,5), (5,4), (5,5).  ζ^9 = 1, ζ^10 = ζ.
		p[0] += a[4] * b[5];   // (4,5) → ζ^9 → slot 0
		p[0] += a[5] * b[4];   // (5,4) → ζ^9 → slot 0
		p[1] += a[5] * b[5];   // (5,5) → ζ^10 → slot 1

		// prod.element[6..8] are still 0 from the default constructor; no reduce() needed.
		return prod;
	}
	

	ringZ9 ringZ9::operator*(const int& right_scalar) const{
		ringZ9 prod;
		
		for(int i = 0; i < 6; i++){
			prod.element[i] = element[i]*right_scalar;
		}
		
		return prod;
	}
	

	ringZ9 ringZ9::operator/(const int& right_scalar) const{
		ringZ9 prod;
		
		if(right_scalar == 0){
			cout << "Zero passed as scalar in ringZ9::operator/(const int& right_scalar)" << endl;
			return prod;
		}
		
		for(int i = 0; i < 6; i++){
			prod.element[i] = element[i]/right_scalar;
		}
		
		return prod;
	}
	
	
	ringZ9 ringZ9::operator=(const ringZ9& right) {
		if (this != &right) {
			for (int i = 0; i < 9; ++i)
				this->element[i] = right.element[i];
			}
		return *this;
	}
	
	
	bool ringZ9::operator==(const ringZ9& right) const{
		ringZ9 diff;
		diff = *this - right;
		return diff.isZero();
	}
	
	bool ringZ9::operator!=(const ringZ9& right) const{
		ringZ9 diff;
		diff = *this - right;
		return !diff.isZero();
	}
	
	
	/* *************************** */
	/* TYPE CONVERSIONS */
	/* *************************** */
	double ringZ9::real_part() const {
                static const double cos_vals[4] = {
                0.766044443118978, 		//cos(2 * M_PI / 9)
                0.17364817766693041, 	//sin(M_PI / 18),
                -0.5, 					//cos(6 pi/9)
                -0.9396926207859083, 	//-cos(M_PI / 9),
                };
                double sum = element[0];
                sum += cos_vals[0] * element[1];
                sum += cos_vals[1] * element[2];
                sum += cos_vals[2] * element[3];
                sum += cos_vals[3] * (element[4]+element[5]);
                return sum;
        }
        
        double ringZ9::imag_part() const {
                static const double sin_vals[4] = {
                        0.6427876096865393,  	// sin(2π/9)
                        0.984807753012208,   	// sin(4π/9)
                        0.8660254037844387,   	// sin(6π/9)
                        0.3420201433256689,   	// sin(8π/9)
                };
                double sum = 0.0;
                sum += sin_vals[0] * element[1];
                sum += sin_vals[1] * element[2];
                sum += sin_vals[2] * element[3];
                sum += sin_vals[3] * (element[4]-element[5]); //The minus sign is due to periodicity
                return sum;
        }
	
	double ringZ9::complexArg() const{
		return atan2(imag_part(), real_part());
	}
	
	
	double ringZ9::abs_val() const{
		return sqrt(real_part()*real_part() + imag_part()*imag_part());
	}
	
	
	double ringZ9::abs_val_sq() const{
		return real_part()*real_part() + imag_part()*imag_part();
	}
	
	
	complex<double> ringZ9::toComplexDouble() const{
		complex<double> z(real_part(),imag_part());
		return z;
	}
	
	
	/* **************************** */
	/* NUMBER THEORETIC */
	/* **************************** */
	// Complex conjugate (Fix E.3, 2026-05).
	//
	// The original code wrote the conjugate into a non-canonical 9-slot
	// array `swap` (placing element[i] at index (8*i)%9), then constructed
	// a ringZ9 — which copied 9 ints AND ran reduce() to fold slots 6..8
	// back into 0..5.  Folding the relations algebraically, the result has
	// the closed form below in the canonical 6-slot basis directly.
	//
	// Derivation: ζ_9^{-i} for i = 1..5 expands via ζ_9^{-1} = ζ_9^8 and
	// the relation ζ_9^6 = -ζ_9^3 - 1, ζ_9^7 = -ζ_9^4 - ζ_9, ζ_9^8 = -ζ_9^5 - ζ_9^2:
	//   ζ_9^0 = 1
	//   ζ_9^{-1} = ζ_9^8 = -ζ_9^2 - ζ_9^5
	//   ζ_9^{-2} = ζ_9^7 = -ζ_9 - ζ_9^4
	//   ζ_9^{-3} = ζ_9^6 = -1 - ζ_9^3
	//   ζ_9^{-4} = ζ_9^5
	//   ζ_9^{-5} = ζ_9^4
	// So conj(Σ a_i ζ_9^i) collected on the canonical {1, ζ_9, ..., ζ_9^5} basis
	// has the coefficients computed directly below.  Verified by ringZ9_unit_test
	// (involution, antihomomorphism, a*a* real, (a*a*)[0] == quad(a)).
	ringZ9 ringZ9::complexConj() const{
		ringZ9 result;
		result.element[0] = element[0] - element[3];
		result.element[1] = -element[2];
		result.element[2] = -element[1];
		result.element[3] = -element[3];
		result.element[4] = element[5] - element[2];
		result.element[5] = element[4] - element[1];
		// element[6..8] remain zero from the default constructor.
		return result;
	}
	
	

	ringZ9 ringZ9::GaloisAut(int k) const{
		k = k%9;
		int swap[9] = {0,0,0,0,0,0,0,0,0};
		
		for (int i = 0; i < 6; i++){
			swap[(((k*i % 9) + 9) % 9)] = swap[(((k*i % 9) + 9) % 9)] + element[i];
		}
		return ringZ9(swap);
	}
	

	int ringZ9::fieldNorm() const{
		ringZ9 prod(1);
		
		for(int i = 1; i < 9; i++){
			if(gcd(i,9) == 1){
				prod = prod*GaloisAut(i);
			}
		}
		
		prod.reduce();
 		return prod.getTerm(0);
	}
	

	int ringZ9::fieldTrace() const{
		ringZ9 sum = GaloisAut(1);
		
		for(int i = 2; i < 9; i++){
			if(gcd(i,9) == 1){
				sum = sum+GaloisAut(i);
			}
		}
 		return sum.getTerm(0);
	}
	
	
	ringZ9 ringZ9::partialFieldNorm() const{
		ringZ9 prod = GaloisAut(2);
		
		for(int i = 4; i < 9; i++){
			if(gcd(i,9) == 1){
				prod = prod*GaloisAut(i);
			}
		}
		return prod;		
	}
	
	int ringZ9::tauFieldNorm() const{
		ringZ9 prod = GaloisAut(1)*GaloisAut(2)*GaloisAut(5);
		
		return prod.getTerm(0);
	}
	

	int ringZ9::weight() const{
		int sum = 0;
		
		for(int i = 0; i < 6; i++){
			sum = sum + abs(element[i]);
		}
		return sum;
	}
	
	int ringZ9::signedWeight() const{
		int sum = 0;
		
		for(int i = 0; i < 6; i++){
			sum = sum + element[i];
		}
		return sum;
	}
	
	int ringZ9::quad() const{
		int sum = 0;
		
		for(int i = 0; i < 6; i++){
			sum = sum + element[i]*element[i];
		}
		
		return sum - element[0]*element[3] - element[1]*element[4] - element[2]*element[5];
	}
	
	
	ringZ9 ringZ9::formalDerivative() const{
		
		int new_element[9] = {0};
		
		for(int i = 0; i < 6; i++){
			new_element[i] = (i+1)*element[i+1];
		}
		
		return ringZ9(new_element);
	}
	
	
	int ringZ9::sdeChi() const{
		if(signedWeight() % 3 != 0) return 0;
		
		ringZ9 test = formalDerivative();
		
		for(int i = 1; i < 6; i++){
			if((test.signedWeight() / i)% 3 != 0 ) return i;
			
			test = (test.formalDerivative())/i;
			
		}
		
		
		return 6;
	}
	
	
	/* ********* */
	/* TESTS */
	/* ********* */
	bool ringZ9::isZero() const{
		
		for(int i = 0; i < 3; i++){
			for(int j = 1; j < 3; j++){
				
				if(element[i + 3*j] !=element[i]){
					return false;
				}
			}
		}
		
		return true;
	}
	
	
	bool ringZ9::isReal() const{
		ringZ9 test = *this - this->complexConj();
		return test.isZero();
	}
	
	
	bool ringZ9::isImag() const{
		ringZ9 test = *this + this->complexConj();
		return test.isZero();
	}
	

	bool ringZ9::isInt() const{
		for(int i = 1; i < 6; i++){
			if(element[i] == 0){
				return false;
			}
		}
		return true;
	}
	
	bool ringZ9::isDivisibleByInt(int k) const{
		for(int i = 0; i < 6; i++){
			if(element[i]%k != 0){
				return false;
			}
		}
		return true;
	}
	
	
	/* ********** */
	/* PRINTS */
	/* ********** */
	void ringZ9::print() const{
		
		if(isZero()){ // prints zero if element equals zero
			cout << "(0)" << endl;
			return;
		}
		
		int i = 0;
		int j = 8;
		
		while(element[i] == 0){ i++;} // i is now the exponent on the first nonzero term
		while(element[j] == 0){	j--;} // j is now the expoent on the last nonzero term
		
		
		if(i == j){
			cout << "(" << element[i] << "z^" << i << ")";
			return;
		}
		
		cout << "(";
		
		for(int k = i; k < j; k++){ 
			if(element[k] != 0){
				cout << element[k] << "z^" << k << " + ";
			}
		}
		
		cout << element[j] << "z^" << j << ")";
	}
	
