// lookup_check.cpp — focused correctness probe for fullX3Enumeration.
// Skips fullDiagEnumeration entirely (which produces 26.5 M candidates and
// risks OOM).  Calls fullX3Enumeration at a hardcoded minQ and reports:
//   1. lookup size
//   2. whether HRSA's V off-diagonal entries (V[0][1], V[0][2], V[1][2]) are
//      present in the output.
//
// Reference data points:
//   - Pre-Fix-P (with A_epsilon bug):  lookup_size=29785 at minQ=2183, OFF-DIAGS MISSING.
//   - Post-Fix-P+R (this binary):       expect lookup_size ≥ 29785, OFF-DIAGS FOUND.
//
// Usage: ./lookup_check [minQ]    (default minQ=2183)

#include "exhaustive_search.h"
#include "cyclotomic_int9.h"
#include <atomic>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <sstream>
#include <string>

using namespace std;

std::atomic<bool> interrupted(false);

static ringZ9 build_ring(const array<int,9>& a){
    int arr[9] = {a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8]};
    return ringZ9(arr);
}

static bool ring_eq(const ringZ9& a, const ringZ9& b){
    for(int k=0; k<6; ++k) if(a.getTerm(k) != b.getTerm(k)) return false;
    return true;
}

static bool find_in(const ringZ9& target, const vector<ringZ9>& vec){
    for(const ringZ9& r : vec) if(ring_eq(target, r)) return true;
    return false;
}

int main(int argc, char** argv){
    int minQ = (argc > 1) ? atoi(argv[1]) : 2183;

    // Read V from /tmp/hrsa_v_numerators.txt
    ifstream fin("/tmp/hrsa_v_numerators.txt");
    if(!fin){ cerr << "Cannot open /tmp/hrsa_v_numerators.txt\n"; return 1; }
    array<array<ringZ9,3>,3> V;
    double theta=0, eps=0;
    int v_exp=-1;
    string line;
    while(getline(fin, line)){
        if(line.empty()) continue;
        if(line[0]=='#'){
            size_t p = line.find("theta=");   if(p!=string::npos) theta = stod(line.substr(p+6));
            p        = line.find("epsilon="); if(p!=string::npos) eps   = stod(line.substr(p+8));
            p        = line.find("v_exp=");   if(p!=string::npos) v_exp = stoi(line.substr(p+6));
            continue;
        }
        istringstream ss(line);
        int i,j,e; array<int,9> a;
        ss>>i>>j>>e; for(int k=0;k<9;++k) ss>>a[k];
        V[i][j] = build_ring(a);
        (void)e;
    }
    cout << "theta=" << theta << " eps=" << eps << " f=" << v_exp << " minQ=" << minQ << "\n";

    vector<ringZ9> lookup;
    fullX3Enumeration(lookup, v_exp, minQ, eps);

    cout << "lookup size: " << lookup.size() << "\n";
    cout << "  V[0][1] (x_2) in lookup: " << (find_in(V[0][1], lookup) ? "FOUND" : "MISSING") << "\n";
    cout << "  V[0][2] (x_3) in lookup: " << (find_in(V[0][2], lookup) ? "FOUND" : "MISSING") << "\n";
    cout << "  V[1][2] (y_3) in lookup: " << (find_in(V[1][2], lookup) ? "FOUND" : "MISSING") << "\n";
    return 0;
}
