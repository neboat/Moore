// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
///<<
static void mm_8_4_256_256(double *Asub,
                           double *Bsub,
                           double *restrict Csub,
                           size_t K) {
// Effect: Compute $\cvrb|Csub|=\cvrb|Csub|+\cvrb|Asub|\cdot{}\cvrb|Bsub|$ where
//  $\cvrb|Asub|$ is an $8\times{}\cvrb|BC|$ matrix stored in column-major order, $32$-byte aligned.
//  $\cvrb|Bsub|$ is an $\cvrb|BC|\times 4$ matrix stored in row-major order, $32$-byte aligned.
//  $\cvrb|Csub|$ is an $8\times{}4$ matrix stored in row-major order, $32$-byte aligned.
// Requires: $\cvrb|Csub|$ does not alias $\cvrb|Asub|$ or $\cvrb|Bsub|$.
/** BEGIN HIDDEN **/
// Usage note: This function can be used as the inner loop of a $\cvrb|BC|\times{}\cvrb|BC|$ matrix multiply.
// Implementation note: This function is coded for AVX (e.g., Intel Sandy Bridge processors).
//  AVX provides 16 vector registers (ymm0 through ymm15), each containing 256 bits (4 doubles).
//  The muladd primitive is be coded using the AVX2 fused multiply-add (but can easily be reverted back: see mm_outer_muladd.c)
/** END HIDDEN **/
  //  We use 8 of the registers to hold the 32 values of $\cvrb|Csub|$.
  __m256d c0 = {0, 0, 0, 0};  // $\cvrb|Csub|_{0,0}, \cvrb|Csub|_{1,1}, \cvrb|Csub|_{2,2}, \cvrb|Csub|_{3,3}$\lilabel{Csubdecls(}
  __m256d c1 = {0, 0, 0, 0};  // $\cvrb|Csub|_{0,2}, \cvrb|Csub|_{1,3}, \cvrb|Csub|_{2,0}, \cvrb|Csub|_{3,1}$
  __m256d c2 = {0, 0, 0, 0};  // $\cvrb|Csub|_{1,0}, \cvrb|Csub|_{0,1}, \cvrb|Csub|_{3,2}, \cvrb|Csub|_{2,3}$
  __m256d c3 = {0, 0, 0, 0};  // $\cvrb|Csub|_{1,2}, \cvrb|Csub|_{0,3}, \cvrb|Csub|_{3,0}, \cvrb|Csub|_{2,1}$
  __m256d c4 = {0, 0, 0, 0};  // $\cvrb|Csub|_{4,0}, \cvrb|Csub|_{5,1}, \cvrb|Csub|_{6,2}, \cvrb|Csub|_{7,3}$
  __m256d c5 = {0, 0, 0, 0};  // $\cvrb|Csub|_{4,2}, \cvrb|Csub|_{5,3}, \cvrb|Csub|_{6,0}, \cvrb|Csub|_{7,1}$
  __m256d c6 = {0, 0, 0, 0};  // $\cvrb|Csub|_{5,0}, \cvrb|Csub|_{4,1}, \cvrb|Csub|_{7,2}, \cvrb|Csub|_{6,3}$
  __m256d c7 = {0, 0, 0, 0};  // $\cvrb|Csub|_{5,2}, \cvrb|Csub|_{4,3}, \cvrb|Csub|_{7,0}, \cvrb|Csub|_{6,1}$\lilabel{Csubdecls)}
/** BEGIN HIDDEN **/
  if (0) {
    _mm_prefetch((void*)&Csub[0*K], _MM_HINT_T0);
    _mm_prefetch((void*)&Csub[1*K], _MM_HINT_T0);
    _mm_prefetch((void*)&Csub[2*K], _MM_HINT_T0);
    _mm_prefetch((void*)&Csub[3*K], _MM_HINT_T0);
    _mm_prefetch((void*)&Csub[4*K], _MM_HINT_T0);
    _mm_prefetch((void*)&Csub[5*K], _MM_HINT_T0);
    _mm_prefetch((void*)&Csub[6*K], _MM_HINT_T0);
    _mm_prefetch((void*)&Csub[7*K], _MM_HINT_T0);
  }
/** END HIDDEN **/
#pragma unroll(4)  ///\lilabel{pragma}
  for (size_t i = 0; i < BC; i++) {  ///\lilabel{loop(}
/** BEGIN HIDDEN **/
    // Don't prefetch everything (this is way too many prefetch instructions).  Also, the hardware prefetcher is likely to work well.
    if (0) _mm_prefetch((void*)&Bsub[i * 4 + BC], _MM_HINT_T2);
/** END HIDDEN **/
    __m256d a0 = *((__m256d*)&(Asub[i * 8]));         // $\cvrb|Asub|_{0,i}, \cvrb|Asub|_{1,i}, \cvrb|Asub|_{2,i}, \cvrb|Asub|_{3,i}$
    __m256d b0 = *((__m256d*)&(Bsub[i * 4]));         // $\cvrb|Bsub|_{i,0}, \cvrb|Bsub|_{i,1}, \cvrb|Bsub|_{i,2}, \cvrb|Bsub|_{i,3}$
    c0 = muladd(a0,  b0,  c0);
    __m256d b0p = _mm256_permute2f128_pd(b0, b0, 1);  // $\cvrb|Bsub|_{i,2}, \cvrb|Bsub|_{i,3}, \cvrb|Bsub|_{i,0}, \cvrb|Bsub|_{i,1}$\lilabel{permuteBsub}
    c1 = muladd(a0,  b0p, c1);
    __m256d a0p = _mm256_permute_pd(a0, 5);           // $\cvrb|Asub|_{1,i}, \cvrb|Asub|_{0,i}, \cvrb|Asub|_{3,i}, \cvrb|Asub|_{2,i}$\lilabel{permuteAsubtop}
    c2 = muladd(a0p, b0,  c2);
    c3 = muladd(a0p, b0p, c3);
    __m256d a1 = *((__m256d*)&(Asub[i * 8 + 4]));     // $\cvrb|Asub|_{4,i}, \cvrb|Asub|_{5,i}, \cvrb|Asub|_{6,i}, \cvrb|Asub|_{7,i}$
    c4 = muladd(a1,  b0,  c4);
    c5 = muladd(a1,  b0p, c5);
    __m256d a1p =  _mm256_permute_pd(a1, 5);          // $\cvrb|Asub|_{5,i}, \cvrb|Asub|_{4,i}, \cvrb|Asub|_{7,i}, \cvrb|Asub|_{6,i}$\lilabel{permuteAsubbot}
    c6 = muladd(a1p, b0,  c6);
    c7 = muladd(a1p, b0p, c7);
  }  ///\lilabel{loop)}
  addstore4(c0, &Csub[0 * K + 0], &Csub[1 * K + 1], &Csub[2 * K + 2], &Csub[3 * K + 3]);  ///\lilabel{addstore(}
  addstore4(c1, &Csub[0 * K + 2], &Csub[1 * K + 3], &Csub[2 * K + 0], &Csub[3 * K + 1]);
  addstore4(c2, &Csub[1 * K + 0], &Csub[0 * K + 1], &Csub[3 * K + 2], &Csub[2 * K + 3]);
  addstore4(c3, &Csub[1 * K + 2], &Csub[0 * K + 3], &Csub[3 * K + 0], &Csub[2 * K + 1]);
  addstore4(c4, &Csub[4 * K + 0], &Csub[5 * K + 1], &Csub[6 * K + 2], &Csub[7 * K + 3]);
  addstore4(c5, &Csub[4 * K + 2], &Csub[5 * K + 3], &Csub[6 * K + 0], &Csub[7 * K + 1]);
  addstore4(c6, &Csub[5 * K + 0], &Csub[4 * K + 1], &Csub[7 * K + 2], &Csub[6 * K + 3]);
  addstore4(c7, &Csub[5 * K + 2], &Csub[4 * K + 3], &Csub[7 * K + 0], &Csub[6 * K + 1]);  ///\lilabel{addstore)}
}
