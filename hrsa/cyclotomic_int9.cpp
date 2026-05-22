// cyclotomic_int9.cpp
//
// As of the 2026-05-13 templated rewrite, ringZ9Base<T> is fully header-only
// (templates require it).  This .cpp file is intentionally minimal — kept so
// the Makefile target list need not change.  The header cyclotomic_int9.h
// defines:
//   template<typename T> class ringZ9Base
//   using ringZ9    = ringZ9Base<int>;
//   using ringZ9Big = ringZ9Base<boost::multiprecision::cpp_int>;
//
// See cyclotomic_int9.h.orig.bak for the pre-templated int-only version.

#include "cyclotomic_int9.h"
