// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
// Break this out into a separate file.
///<<----------------------------------------------------------------------
static inline __m256d muladd(__m256d x, __m256d y, __m256d z) {
// Effect: Return vector $\cvrb|v|=\cvrb|z|+\cvrb|x|\cvrb|y|$, i.e. where $\cvrb|v|_i=\cvrb|z|_i+\cvrb|x|_i{}\cvrb|y|_i$
//   for $0\leq{}i<4$.
  if (0) {
    return _mm256_fmadd_pd(x, y, z);
  } else {
    return _mm256_add_pd(z, _mm256_mul_pd(x, y));
  }
}
///>>----------------------------------------------------------------------
