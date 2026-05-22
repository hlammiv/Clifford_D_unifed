// decompose_cli.cpp
//
// Standalone CLI front-end for the Clifford+D decomposition routine in
// decompose.cpp.  Reads a JSON description of a 3x3 unitary V over
// Z[zeta_9, 1/3^f] from stdin, calls decompose(V, false), and writes a
// JSON result to stdout.
//
// JSON input schema (most direct mapping from ringZ9chi's internal layout):
//
//   {
//     "f": <int>,                      // common denominator exponent: 3^f
//     "V": [ [ <entry>, <entry>, <entry> ],   // row 0
//            [ <entry>, <entry>, <entry> ],   // row 1
//            [ <entry>, <entry>, <entry> ] ]  // row 2
//   }
//
// where each <entry> is a JSON array of exactly 6 integers giving the
// coefficients on the canonical Z-basis {1, zeta_9, zeta_9^2, zeta_9^3,
// zeta_9^4, zeta_9^5} of the numerator in Z[zeta_9].  This matches the
// reduced canonical form returned by ringZ9::getStdArray() / ringZ9chi::
// getStdArray() — entries 6,7,8 are always zero after reduce(), so the
// 6-int form is the natural external schema.
//
// The actual matrix entry is sum_i (coeffs[i] * zeta_9^i) / 3^f.
// All entries share the same denominator exponent f.  This is consistent
// with how matrices come out of HRSA / diagSearch in this codebase.
//
// Output JSON schema:
//
//   {
//     "success": <bool>,
//     "D_count": <int>,
//     "sde_chi_initial": <int>,
//     "sde_chi_final": <int>,
//     "syllables": [
//       {"a0":int, "a1":int, "a2":int, "eps":int, "delta":int, "has_H":bool},
//       ...
//     ],
//     "trailing_clifford": {
//       "f": <int>,                 // denominator exponent of the residual
//       "V": [[ [6 ints]×3 ]×3]     // residual monomial Clifford+D matrix
//     }
//   }
//
// `trailing_clifford` is the residual matrix after peeling: for the general
// path it is monomial at sde_chi=0; for the diagonal fast-path it is the
// unpeeled input V (and sde_chi_final equals max sde_chi of the diagonal).
// Multiplying the syllables in order against `trailing_clifford` reproduces
// the input V exactly in the ring (see DecompResult docstring in decompose.h).
//
// Exit codes:
//   0  success (regardless of decomposition success/failure flag — those
//      are reported in the JSON; non-zero exit is reserved for tooling
//      errors so a Python wrapper can distinguish "ran fine" from "broke")
//   1  JSON parse error
//   2  schema/dimension error
//
// Build: see Makefile target `decompose_tool`.

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include <cstdlib>
#include <array>
#include <boost/multiprecision/cpp_int.hpp>

#include "cyclotomic_int9.h"
#include "Z9chi.h"
#include "decompose.h"
#include "decompose_impl.h"

using cpp_int = boost::multiprecision::cpp_int;

// Emit the cpp_int instantiation of decompose() and friends in this TU.
template DecompResultBase<cpp_int> decompose<cpp_int>(Mat3Base<cpp_int> V, bool quiet);
template int sdeChiFull<cpp_int>(const ringZ9chiBase<cpp_int>& x);
template int sdeChiZ9<cpp_int>(ringZ9Base<cpp_int> a);
template Mat3Base<cpp_int> buildUnitary<cpp_int>(const std::array<ringZ9chiBase<cpp_int>,3>& u);

// ---------------------------------------------------------------------------
//  Minimal hand-written JSON reader.
//
//  Supports: objects, arrays, signed integers, true/false, strings (used
//  only as object keys here), whitespace.  Does NOT support floats,
//  escapes inside strings beyond \\ and \", null, scientific notation,
//  trailing commas.  That's sufficient for the schema above.
// ---------------------------------------------------------------------------

struct JsonValue {
    enum Type { OBJECT, ARRAY, INTEGER, BOOLEAN, STRING } type;
    // Big-int storage: SK-assembled unitaries can have coefficients well
    // outside int64 range (O(3^f) for f ≥ 40).  We parse all numeric literals
    // into cpp_int and downcast as needed at each use site.
    cpp_int int_val = 0;
    bool bool_val = false;
    std::string str_val;
    std::vector<std::pair<std::string, JsonValue>> obj_val;
    std::vector<JsonValue> arr_val;
};

struct JsonParser {
    const std::string& s;
    size_t i = 0;
    explicit JsonParser(const std::string& src) : s(src) {}

    void skip_ws() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("JSON parse error at offset " +
                                 std::to_string(i) + ": " + msg);
    }

    char peek() {
        skip_ws();
        if (i >= s.size()) fail("unexpected end of input");
        return s[i];
    }

    void expect(char c) {
        skip_ws();
        if (i >= s.size() || s[i] != c) fail(std::string("expected '") + c + "'");
        ++i;
    }

    JsonValue parse_value() {
        skip_ws();
        if (i >= s.size()) fail("unexpected end of input");
        char c = s[i];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_int();
        fail(std::string("unexpected character '") + c + "'");
    }

    JsonValue parse_object() {
        JsonValue v;
        v.type = JsonValue::OBJECT;
        expect('{');
        skip_ws();
        if (i < s.size() && s[i] == '}') { ++i; return v; }
        while (true) {
            skip_ws();
            JsonValue key = parse_string();
            expect(':');
            JsonValue val = parse_value();
            v.obj_val.emplace_back(key.str_val, std::move(val));
            skip_ws();
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            if (i < s.size() && s[i] == '}') { ++i; break; }
            fail("expected ',' or '}'");
        }
        return v;
    }

    JsonValue parse_array() {
        JsonValue v;
        v.type = JsonValue::ARRAY;
        expect('[');
        skip_ws();
        if (i < s.size() && s[i] == ']') { ++i; return v; }
        while (true) {
            v.arr_val.push_back(parse_value());
            skip_ws();
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            if (i < s.size() && s[i] == ']') { ++i; break; }
            fail("expected ',' or ']'");
        }
        return v;
    }

    JsonValue parse_string() {
        JsonValue v;
        v.type = JsonValue::STRING;
        expect('"');
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char n = s[i + 1];
                if (n == '"' || n == '\\') { v.str_val.push_back(n); i += 2; continue; }
                fail("unsupported escape");
            }
            v.str_val.push_back(s[i]);
            ++i;
        }
        expect('"');
        return v;
    }

    JsonValue parse_int() {
        JsonValue v;
        v.type = JsonValue::INTEGER;
        skip_ws();
        size_t start = i;
        if (i < s.size() && s[i] == '-') ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        if (start == i || (s[start] == '-' && start + 1 == i))
            fail("malformed integer");
        // cpp_int parses arbitrary-precision integers from a string.
        std::string digits = s.substr(start, i - start);
        std::istringstream iss(digits);
        iss >> v.int_val;
        if (iss.fail())
            fail("integer literal failed to parse");
        return v;
    }

    JsonValue parse_bool() {
        JsonValue v;
        v.type = JsonValue::BOOLEAN;
        skip_ws();
        if (s.compare(i, 4, "true") == 0) { v.bool_val = true; i += 4; return v; }
        if (s.compare(i, 5, "false") == 0) { v.bool_val = false; i += 5; return v; }
        fail("expected true or false");
    }
};

// Helpers to fish keys out of an OBJECT JsonValue.
static const JsonValue* find_key(const JsonValue& obj, const std::string& key) {
    if (obj.type != JsonValue::OBJECT) return nullptr;
    for (const auto& kv : obj.obj_val) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

// Process one JSON request and emit one JSON response on `out` (one line, no
// trailing newline — caller is responsible for terminator). Returns 0 on
// success, 1 for parse failure, 2 for schema failure.
//
// Extracted from main() so the same code path serves both the legacy slurp-
// stdin mode and the new --persistent mode (one request per stdin line).
static int process_request(const std::string& input, std::ostream& out) {
    JsonValue root;
    try {
        JsonParser parser(input);
        root = parser.parse_value();
    } catch (const std::exception& e) {
        std::cerr << "decompose_tool: " << e.what() << std::endl;
        return 1;
    }

    if (root.type != JsonValue::OBJECT) {
        std::cerr << "decompose_tool: top-level JSON must be an object" << std::endl;
        return 2;
    }

    const JsonValue* fp = find_key(root, "f");
    const JsonValue* Vp = find_key(root, "V");
    if (!fp || fp->type != JsonValue::INTEGER) {
        std::cerr << "decompose_tool: missing or non-integer 'f' field" << std::endl;
        return 2;
    }
    if (!Vp || Vp->type != JsonValue::ARRAY || Vp->arr_val.size() != 3) {
        std::cerr << "decompose_tool: 'V' must be a length-3 array of rows" << std::endl;
        return 2;
    }

    // 'f' (denominator exponent on 3) is always a small int.
    int f = fp->int_val.convert_to<int>();

    // Build Mat3Big: cpp_int storage so we can hold the O(3^f) coefficients
    // that come out of SK-assembled exact unitaries.  Each entry's denominator
    // exponent is the shared 'f' on input; the ringZ9chi constructor will
    // normalize() and may pull out common factors of 3.
    Mat3Big V;
    for (int r = 0; r < 3; ++r) {
        const JsonValue& row = Vp->arr_val[r];
        if (row.type != JsonValue::ARRAY || row.arr_val.size() != 3) {
            std::cerr << "decompose_tool: row " << r
                      << " must be a length-3 array" << std::endl;
            return 2;
        }
        for (int c = 0; c < 3; ++c) {
            const JsonValue& entry = row.arr_val[c];
            if (entry.type != JsonValue::ARRAY || entry.arr_val.size() != 6) {
                std::cerr << "decompose_tool: V[" << r << "][" << c
                          << "] must be a length-6 integer array" << std::endl;
                return 2;
            }
            // ringZ9chiBig constructor takes a 9-cpp_int array; entries 6..8
            // are zero in the canonical reduced basis.
            cpp_int arr9[9];
            for (int k = 0; k < 9; ++k) arr9[k] = cpp_int(0);
            for (int k = 0; k < 6; ++k) {
                const JsonValue& coef = entry.arr_val[k];
                if (coef.type != JsonValue::INTEGER) {
                    std::cerr << "decompose_tool: V[" << r << "][" << c
                              << "][" << k << "] must be an integer" << std::endl;
                    return 2;
                }
                arr9[k] = coef.int_val;
            }
            V.m[r][c] = ringZ9chiBig(arr9, f);
        }
    }

    // Run the decomposition on the cpp_int instantiation.
    DecompResultBig res = decompose<cpp_int>(V, /*quiet=*/true);

    // Emit JSON.  We hand-format to keep the output deterministic.
    out << "{";
    out << "\"success\": " << (res.success ? "true" : "false");
    out << ", \"D_count\": " << res.D_count;
    out << ", \"sde_chi_initial\": " << res.sde_chi;
    out << ", \"sde_chi_final\": " << res.sde_chi_final;
    out << ", \"syllables\": [";
    for (size_t k = 0; k < res.steps.size(); ++k) {
        const GateStep& g = res.steps[k];
        if (k) out << ", ";
        out << "{"
            << "\"a0\": " << g.a0
            << ", \"a1\": " << g.a1
            << ", \"a2\": " << g.a2
            << ", \"eps\": " << g.eps
            << ", \"delta\": " << g.delta
            << ", \"has_H\": " << (g.has_H ? "true" : "false")
            << "}";
    }
    out << "]";

    // Trailing residual: emit the same {f, V[3][3][6]} shape we accept on input.
    int trail_f = 0;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!res.trailing_clifford.m[r][c].isZero()) {
                trail_f = res.trailing_clifford.m[r][c].getExp();
                goto found_trail_f;
            }
        }
    }
    found_trail_f:

    out << ", \"trailing_clifford\": {";
    out << "\"f\": " << trail_f;
    out << ", \"V\": [";
    for (int r = 0; r < 3; ++r) {
        if (r) out << ", ";
        out << "[";
        for (int c = 0; c < 3; ++c) {
            if (c) out << ", ";
            std::array<cpp_int, 6> a = res.trailing_clifford.m[r][c].getStdArray();
            out << "[";
            for (int k = 0; k < 6; ++k) {
                if (k) out << ", ";
                out << a[k];
            }
            out << "]";
        }
        out << "]";
    }
    out << "]}";

    out << "}";  // no trailing endl — caller adds delimiter

    return 0;
}


int main(int argc, char* argv[]) {
    // Dispatch: --persistent enters a loop reading one JSON request per
    // stdin line and emitting one JSON response per line. The static
    // canonical_lookup BFS table is built lazily on first decompose<T>()
    // call and persists for the life of the process — saving the ~15s
    // build cost on every subsequent request (key win per audit
    // decompose_optimization_audit.md).
    bool persistent = false;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--persistent" || a == "-p") {
            persistent = true;
        } else if (a == "--help" || a == "-h") {
            std::cerr << "Usage: decompose_tool [--persistent]\n"
                      << "  default: read one JSON from stdin, emit one JSON, exit\n"
                      << "  --persistent: loop reading one JSON-per-line from stdin\n";
            return 0;
        } else {
            std::cerr << "decompose_tool: unknown arg: " << a << std::endl;
            return 2;
        }
    }

    if (persistent) {
        std::string line;
        // One JSON request per line; one JSON response per line.
        // EOF on stdin → clean shutdown.
        while (std::getline(std::cin, line)) {
            // Skip blank lines (keep-alive friendly).
            if (line.empty()) continue;
            int rc = process_request(line, std::cout);
            if (rc != 0) {
                // On parse/schema error, emit an error JSON so the caller can
                // match the request/response 1-to-1 and not block reading.
                std::cout << "{\"success\": false, \"D_count\": -1, "
                             "\"error\": \"parse_or_schema_failure\", "
                             "\"returncode\": " << rc << "}";
            }
            std::cout << "\n";
            std::cout.flush();
        }
        return 0;
    }

    // Legacy: slurp stdin, process once, exit. Backward compatible with
    // all existing callers (sk_reduce.py, zeta9_compile.py, manual use).
    std::ostringstream buf;
    buf << std::cin.rdbuf();
    int rc = process_request(buf.str(), std::cout);
    std::cout << std::endl;
    return rc;
}
