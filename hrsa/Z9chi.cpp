// Z9chi.cpp
//
// As of the 2026-05-13 templated rewrite, ringZ9chiBase<T> is fully header-only
// (templates require it).  This .cpp file is intentionally minimal — kept so
// the Makefile target list need not change.  The header Z9chi.h defines:
//   template<typename T> class ringZ9chiBase
//   using ringZ9chi    = ringZ9chiBase<int>;
//   using ringZ9chiBig = ringZ9chiBase<boost::multiprecision::cpp_int>;
//
// See Z9chi.cpp.orig.bak for the pre-templated int-only version.

#include "Z9chi.h"
