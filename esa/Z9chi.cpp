#include "Z9chi.h"

	/* *********************** */
	/* CONSTRUCTORS  */
	/* *********************** */
	ringZ9chi::ringZ9chi(int arr[9], int denom_exp){

		exp = denom_exp;
		
		for(int i = 0; i < 9; ++i) {
			element[i] = arr[i];
		}
		
		normalize();
	}
	
	// Option to conctruct an element of Z[zeta_9,1/3] using an 
	// element of Z[zeta_9] (class ringZ9) as the numerator
	ringZ9chi::ringZ9chi(ringZ9 numer, int denom_exp){
		exp = denom_exp;
		
		for(int i = 0; i < 6; i++){
			element[i] = numer.getTerm(i);
		}
		cancel();
	}
	
	
	// defining an instance of ringZ9chi with no argument returns the zero element.
	ringZ9chi::ringZ9chi(){

		exp = 0;
		for(int i = 0; i < 9; i++){
				element[i] = 0;
		}
	}
	
	/* ******* */
	/* INFO */
	/* ******* */
	// gets the coefficient on the i-th term
	int ringZ9chi::getTerm(int i) const {
		if(i > 8 || i < 0){
			return 0;
		} else {
			return element[i];
		}	
	}
	
	// gets the exponent on the 1/3
	int ringZ9chi::getExp() const {
		return exp;
	}
	
	ringZ9 ringZ9chi::getNumerator() const{
		
		int copy[9] = {0,0,0,0,0,0,0,0,0};
		for(int i = 0; i < 6; i++){
			copy[i] = element[i];
		}
		
		return ringZ9(copy);
	}
	
	// returns a std::array suitable for matrix computations.
	array<int, 6> ringZ9chi::getStdArray() const{
		
		array<int, 6> arr;
		
		for(int i = 0; i < 6; i++){
			arr[i] = element[i];
		}
		return arr;
	}
	
	
	/* ********************* */
	/* SIMPLIFICATION */
	/* ********************* */
	/* If possible, simplifies expression into a UNIQUE form.
		
	After testing to make sure ringZ9chi element is nonzero, the function calls reduce()
	which rewrites the numerator of the ringZ9chi element so that the last three array 
	entries are zero.
	
	Then, the function cancel() is called.
	If each coeff is divisible by 3 and exp > 1, then each coeff is divided
	by 3, and exp (if positive) is decremented by 1. This corresponds to cancelling
	factors of 3 in both the numerator and denominator. This process
	is repeated until there are no 3's left to cancel out.
	
	If z is a primitive 9th rou, then every element of Z[z] may be 
	written UNIQUELY as a Z-linear combo of z^i's where the i's range from
	0 to 5. This follows from the fact that for any primitive p^2-th rou z, the 
	rous of the form z^i for i=0,1,...,p(p-1) - 1 form a Q-basis of Q(z). Also,	
	an element of Z[z] is divisible by 3 exactly when all coeffs are divisble by 3,
	so this function truly gives a unique form. In particular, testing equality between
	two 'normalized' ringZ9chi elements now equates to checking that the variables 'element'
	and 'exp' are identical. */
	void ringZ9chi::normalize(){


		// If the ringZ9chi element is a sum of minimal vanishing sums of rou,
		// then the element is simply zeroed out.
		if(isZero()){
			exp = 0;
			for(int i = 0; i < 9; i++){
				element[i] = 0;
			}
			return;
		}

		reduce();
		cancel();
	}


	// We take advantage of the min. van. sums of 9th roots of unity to 
	// rewrite the numerator, so that the last 3 coeffs become 0
/*	void ringZ9chi::reduce(){

		for(int i = 6; i < 9; i++){
			for(int j = 1; j < 3; j++){
				element[(i + 3*j)%9] = element[(i + 3*j)%9] - element[i];
			}
		}
		element[6] = 0;
		element[7] = 0;
		element[8] = 0;
	}
*/
	void ringZ9chi::reduce() {
    	// Cache the values of element[6], element[7], element[8]
    		int e6 = element[6];
    		int e7 = element[7];
    		int e8 = element[8];

    	// Apply subtraction in a tight, unrolled loop
    		element[0] -= e6; // (6 + 3*1) % 9 = 0
    		element[3] -= e6; // (6 + 3*2) % 9 = 3

    		element[1] -= e7; // (7 + 3*1) % 9 = 1
    		element[4] -= e7; // (7 + 3*2) % 9 = 4

    		element[2] -= e8; // (8 + 3*1) % 9 = 2
    		element[5] -= e8; // (8 + 3*2) % 9 = 5

    // Zero out the last three elements
    		element[6] = 0;
    		element[7] = 0;
    		element[8] = 0;
	}
	
	// Successively cancels 3's showing up in both numerator and denominator
	// 'unfinshed' becomes false when we run out of 3's in numerator
	// likewise, we run out of 3's in denominator when exp becomes zero.
	void ringZ9chi::cancel(){
		bool unfinished = true;
		
		while(unfinished && exp > 0){
			for(int i = 0; i < 6; i++){
				if(element[i] % 3 != 0){
					unfinished = false;
					break; // found a coeff indivisible by 3. No normalization needed
				}
			}
			
			// carries out division by 3 if all coeffs are divisble by 3
			if(unfinished){
				exp--;
				for(int i = 0; i < 6; i++){
					element[i] = element[i]/3;
				}
			}
		}	
	}
	
	
	// multiplies each term by an integer scalar
	void ringZ9chi::scalar_mult(int scalar){
		for(int i = 0; i < 9; i++){
			element[i] = scalar*element[i];
		}
		cancel();
	}
	
	
	/* ***************** */
	/* OPERATIONS */
	/* ***************** */	
	ringZ9chi ringZ9chi::operator+(const ringZ9chi& right) const{
		ringZ9chi sum;
		
		if(exp < right.exp){
			sum.exp = right.exp;
		
			for(int i = 0; i < 6; i++){
				sum.element[i] = pow(3,right.exp - exp)*element[i] + right.element[i];
			}
		
		} else if(exp > right.exp){
			sum.exp = exp;
		
			for(int i = 0; i < 6; i++){
				sum.element[i] = element[i] + pow(3,exp - right.exp)*right.element[i];
			}
		} else {
			for(int i = 0; i < 6; i++){
				sum.exp = exp;
				sum.element[i] = element[i] + right.element[i];
			}
		}
	
	// We can assume the summands have passed through reduce(), so we needed
	// only pass the sum through cancel().
	sum.cancel();
	return sum;
	}
	
	
	ringZ9chi ringZ9chi::operator-(const ringZ9chi& right) const{
			ringZ9chi diff;
		
		if(exp < right.exp){
			diff.exp = right.exp;
		
			for(int i = 0; i < 6; i++){
				diff.element[i] = pow(3,right.exp - exp)*element[i] - right.element[i];
			}
		
		} else if(exp > right.exp){
			diff.exp = exp;
		
			for(int i = 0; i < 6; i++){
				diff.element[i] = element[i] - pow(3,exp - right.exp)*right.element[i];
			}
		} else {
			for(int i = 0; i < 6; i++){
				diff.element[i] = element[i] - right.element[i];
			}
		}
		
	// We can assume the summands have passed through reduce(), so we needed
	// only pass the difference through cancel().
	diff.cancel();
	return diff;
	}
	
	ringZ9chi ringZ9chi::operator*(const ringZ9chi& right) const{
		ringZ9chi prod;
		
		// multiplying denomnators
		prod.exp = exp + right.exp;
		
		// multiplying numerators
		for(int i = 0; i <9; i++){
			for(int j = 0; j < 9; j++)
				prod.element[(i + j)%9] = prod.element[(i + j)%9] + element[i]*right.element[j];
		}
		
		prod.normalize();
		return prod;
	}

	
	ringZ9chi ringZ9chi::operator=(const ringZ9chi& right) {
		if (this != &right) {
			for (int i = 0; i < 9; ++i){
				this->element[i] = right.element[i];
				this->exp = right.exp;
			}
		}
		return *this;
	}
	
	
	// Defining notion of equality in this ringZ9chi.
	bool ringZ9chi::operator==(const ringZ9chi& right) const{
		ringZ9chi diff;
		diff = *this - right;
		return diff.isZero();
	}
	
	bool ringZ9chi::operator!=(const ringZ9chi& right) const{
		ringZ9chi diff;
		diff = *this - right;
		return !diff.isZero();
	}
	

	/* *************************** */
	/* TYPE CONVERSIONS */
	/* *************************** */
	double ringZ9chi::real_part() const{
		double sum = 0;
		
		for(int i = 0; i < 6; i++){
			sum = sum + cos(2*M_PI*i/9)*element[i];
		}
		return sum/pow(3,exp);
	}
	
	double ringZ9chi::imag_part() const{
		double sum = 0;
		
		
		for(int i = 0; i < 6; i++){
			sum = sum + sin(2*M_PI*i/9)*element[i];
		}
		return sum/pow(3,exp);
	}
	
	double ringZ9chi::abs_val() const{
		return sqrt(real_part()*real_part() + imag_part()*imag_part());
	}
	
	double ringZ9chi::abs_val_sq() const{
		return real_part()*real_part() + imag_part()*imag_part();
	}
	
	complex<double> ringZ9chi::toComplexDouble() const{
		complex<double> z(real_part(),imag_part());
		return z;
	}
	
	/* **************************** */
	/* NUMBER THEORETIC */
	/* **************************** */
	ringZ9chi ringZ9chi::complexConj() const{
		
		int swap[9] = {0,0,0,0,0,0,0,0,0};
		
		for (int i = 0; i < 6; i++){
			swap[(8*i)%9] = element[i];
		}
		return ringZ9chi(swap,exp);
	}
	
	// A Galois automorphism of Q(zeta_9)/Q raises the exponent on each rou
	// to the power of k where gcd(k,9) = 1.
	ringZ9chi ringZ9chi::GaloisAut(int k) const{
		k = k%9;
		int swap[9] = {0,0,0,0,0,0,0,0,0};
		
		for (int i = 0; i < 6; i++){
			swap[(((k*i % 9) + 9) % 9)] = swap[(((k*i % 9) + 9) % 9)] + element[i];
		}
		return ringZ9chi(swap,exp);
	}
	
	// The field norm is the product of all Galois conjugates
	double ringZ9chi::fieldNorm() const{
		ringZ9chi prod = *this;
		
		for(int i = 2; i < 9; i++){
			if(gcd(i,9) == 1){
				prod = prod*GaloisAut(i);
			}
		}
		
		prod.reduce();
 		return prod.getTerm(1);
	}
	
	// The field trace is the sum among all Galois conjugates
	double ringZ9chi::fieldTrace() const{
		ringZ9chi sum = *this;
		
		for(int i = 1; i < 9; i++){
			if(gcd(i,9) == 1){
				sum = sum+GaloisAut(i);
			}
		}
 		return sum.getTerm(1);
	}
	
	/* ********* */
	/* TESTS */
	/* ********* */
	// We check to verify whether the element is zero. This amounts to
	// checking whether the ringZ9chi element is a sum of minimal vanish sums of rou's
	//
	// WARNING: Not all calls made to isZero() are guaranteed to have passed through
	// normalize(). Therefore, the for loops must loop through all array entries.
	bool ringZ9chi::isZero() const{
		
		for(int i = 0; i < 3; i++){
			for(int j = 1; j < 3; j++){
				
				if(element[i + 3*j] !=element[i]){
					return false;
				}
			}
		}
		
		return true;
	}
	
	bool ringZ9chi::isReal() const{
		ringZ9chi test = *this - this->complexConj();
		return test.isZero();
	}
	
	bool ringZ9chi::isImag() const{
		ringZ9chi test = *this + this->complexConj();
		return test.isZero();
	}

	
	bool ringZ9chi::isRational() const{
		for(int i = 1; i < 6; i++){
			if(element[i] == 0){
				return false;
			}
		}
		return true;
	}
	
	bool ringZ9chi::isInt() const{
		
		if(isRational() && element[0]% static_cast<int>(pow(3,exp)) == 0){
			return true;
		}
		return false;
	}
	
	bool ringZ9chi::isCycInt() const{
		return exp == 0;
	}

	/* ********** */
	/* PRINTS */
	/* ********** */
	// Prints out the element as a Z-linear combo of ninth rous
	// Here, "z" denotes the primitive 9th rou e^(2*i*pi/9)
	void ringZ9chi::print() const{
		
		if(isZero()){ // prints zero if element equals zero
			cout << "(0)";
			return;
		}
		
		if(exp != 0){
			cout << "(1/3^" << exp << ")"; 
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
