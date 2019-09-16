// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
/* outer product matrix multiply */

#define _GNU_SOURCE
#define NATIVE_AVX 1
#if NATIVE_AVX
#include <immintrin.h>
#else
#include <avxintrin_emu.h>
#endif
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cilk/reducer.h>

void print_matrix(const double *A, size_t X, size_t Y, size_t stride, bool rowmajor) {
  printf(" matrix is %ld by %ld stride %ld%s\n", X, Y, stride, rowmajor?"":" colmajor");
  for (size_t i = 0; i < X; i++) {
    printf("[");
    for (size_t j = 0; j < Y; j++) {
      if (j > 0) printf(" ");
      double V = rowmajor ? A[i*stride+j] : A[j*stride+i];
      printf("%15g", V);
    }
    printf("]\n");
  }
}

void mm_naive(const   double *A,  // XxY matrix in row-major order with stride strideA
              const   double *B,  // YxZ matrix in row-major order with stride strideB
              /*out*/ double * __restrict__ C,  // XxZ matrix in row-major order with stride strideC
              size_t X, size_t Y, size_t Z, size_t strideA, size_t strideB, size_t strideC) {
  for (size_t i = 0; i < X; i++) {
    for (size_t j = 0; j < Y; j++) {
      for (size_t k = 0; k < Z; k++) {
        double Av = A[i*strideA + j];
        double Bv = B[j*strideB + k];
        assert(i*strideC + k < X*strideC);
        double Cv = C[i*strideC + k];
        C[i*strideC + k] = Cv + Av * Bv;
      }
    }
  }
}

#include "mm_outer_muladd.c"
#include "mm_outer_addstore.c"
// BC is typically 256, it's the size of the small matrix that will operate on.
#define logBC 8
#define BC (1 << logBC)

static void print4(__m256d v) __attribute__((unused));
static void print4(__m256d v) {
  struct my4 m;
  *((__m256d*)&m)=v;
  for (int i = 0; i < 4; i++) {
    if (i > 0) printf(" ");
    printf("%f", m.t[i]);
  }
  printf("\n");
}
static double geti(__m256d v, int i) __attribute__((unused));
static double geti(__m256d v, int i) {
  assert(0 <= i && i < 4);
  struct my4 m;
  *((__m256d*)&m)=v;
  return m.t[i];
}

#include "mm_outer_8_4_256_256.c"

void test1(void) {
  double C[8][4];
  double C2[8][4];
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 4; j++) C2[i][j]= C[i][j]=i*4+j;
  double A[BC][8];   // column-major order
  double AT[8][BC];  //  row-major order
  //  printf("test1: C starts as\n"); print_matrix(&C[0][0], 8, 4, 4, true);
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < BC; j++)
      AT[i][j] = A[j][i]=i*10000.0 + j;
  double B[BC][4];
  for (int i = 0; i < BC; i++)
    for (int j = 0; j < 4; j++)
      B[i][j]=1e8 + i*1e4 +j;
  mm_naive(&AT[0][0], &B[0][0], &C2[0][0], 8, BC, 4, BC, 4, 4);
  mm_8_4_256_256(&A[0][0], &B[0][0], &C[0][0], 4);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 4; j++) {
      assert(C[i][j] == C2[i][j]);
    }
  }
  if (0) {
    printf("result:\n");
    print_matrix(&C[0][0], 8, 4, 4, true);
    printf("+=\n");
    print_matrix(&A[0][0], BC, 8, BC, false);
    printf("*\n");
    print_matrix(&B[0][0], BC, 4, 4, true);
  }
}

void *malloc_aligned(size_t size) {
  void *p;
  int r = posix_memalign(&p, 64, size);
  assert(0 == r);
  return p;
}
// Bradley's notebook has a figure for how these work.
///<<----------------------------------------------------------------------
static inline double *A_(double *AT, size_t i, size_t j) {
  return AT + ((i >> 3) << (3 + logBC)) + (j << 3) + (i & 7);
}

static inline double *B_(double *BT, size_t i, size_t j) {
  return BT + ((j >> 2) << (2 + logBC)) + (i << 2) + (j & 3);
}

///>>----------------------------------------------------------------------
#define SLOW_VERIFY 0

static bool close_enough(double A, double B) {
  A = A < 0 ? -A : A;
  B = B < 0 ? -B : B;
  if (A < B) {
    double T = A;
    A = B;
    B = T;
  }
  double diff = A-B;
  if (!(diff >= 0)) fprintf(stderr, "A = %f, B = %f\n", A, B);
  assert(diff >= 0);
  return (diff < 0.01) || (diff/A < 0.00001);
}

void abort_if_too_different(double A, double B, size_t off) {
  bool is_ok = close_enough(A, B);
  if (!is_ok) printf("A=%f B=%f i=%ld\n", A, B, off);
  assert(is_ok);
}

///<<----------------------------------------------------------------------
__thread double *AT = NULL;
__thread double *BT = NULL;

#define DEBUG_TMP_MALLOC 0

struct TmpStack {
  double *T;
  size_t start;
#if DEBUG_TMP_MALLOC
  size_t size;
#endif
};

/* __thread struct TmpStack TS = { NULL, 0, 0 }; */

void identity_tmp_holder(void *reducer __attribute__((unused)), void *view) {
  ((struct TmpStack*)view)->T = NULL;
}

void reduce_tmp_holder(void *reducer __attribute__((unused)),
                       void *l __attribute__((unused)),
                       void *r __attribute__((unused))) {
}

void destroy_tmp_holder(void *reducer __attribute__((unused)),
                        void *view /* __attribute__((unused)) */) {
  /* free(((struct TmpStack*)view)->T); */
  struct TmpStack *viewts = (struct TmpStack*)view;
  /* if (NULL == TS.T) { */
  /*   TS = *viewts; */
  /* } else if (TS.size < viewts->size) { */
  /*   free(TS.T); */
  /*   TS = *viewts; */
  /* } else { */
    free(viewts->T);
  /* } */
}

CILK_C_DECLARE_REDUCER(struct TmpStack) tmp_stack =
  CILK_C_INIT_REDUCER(struct TmpStack,
                      reduce_tmp_holder,
                      identity_tmp_holder,
                      destroy_tmp_holder,
                      {NULL, 0,
#if DEBUG_TMP_MALLOC
                          0
#endif
                          });
/* __thread double *TT = NULL; */
/* __thread size_t TTn = 0; */

static inline void mm_256_256(
    const double *A, /* A[BC][BC] row-major order with stride of Astride. */
    size_t Astride,
    const double *B, /* B[BC][BC] row-major order with stride of Bstride. */
    size_t Bstride,
    /*out*/ double *restrict C, /* C[BC][BC] row-major order with stride of stride. */
    size_t Cstride) {
  // Effect: Compute C = C + A \cdot B$
  //  where A, B, and C are BC\times BC matrices.

  // AT is a temporary BC\times BC array.
  // AT is blocked with address calculations.
  if (AT == NULL) AT = malloc_aligned(sizeof(*A) * BC * BC);  ///\lilabel{createAT}
  if (BT == NULL) BT = malloc_aligned(sizeof(*B) * BC * BC);  ///\lilabel{createBT}
  for (size_t i = 0; i < BC; i++) {         ///\lilabel{blockAB(}
    for (size_t j = 0; j < BC; j++) {
      *(A_(AT, i, j)) = A[i * Astride + j];
      *(B_(BT, i, j)) = B[i * Bstride + j];
    }
  }                                     ///\lilabel{blockAB)}
  for (size_t i = 0; i < BC; i += 8) {      ///\lilabel{computeC(}
    for (size_t k = 0; k < BC; k += 4) {
      ///>>----------------------------------------------------------------------
      // printf("Mult AT[%d,*] by BT[*,%d] into C[%d][%d]\n", i, k, i, k);
      ///<<----------------------------------------------------------------------
      mm_8_4_256_256(A_(AT, i, 0), B_(BT, 0, k), C + i * Cstride + k, Cstride);
    }
  }                                     ///\lilabel{computeC)}
}
///>>----------------------------------------------------------------------

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) +1e-6*(end->tv_usec - start->tv_usec);
}

void test2(int N) {
  double *C  = malloc(N*N*sizeof(*C));
  double *C2 = malloc(N*N*sizeof(*C2));
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      C2[i*N+j] = C[i*N+j] = 0;
  double *A  = malloc(N*N*sizeof(*A));
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      A[i*N+j] = i*10000.0 + j;
  double *B  = malloc(N*N*sizeof(*B));
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      B[i*N+j]=1e8 + i*1e4 +j;
  mm_naive(A, B, C2, N, N, N, N, N, N);
  struct timeval start, end;
  gettimeofday(&start, NULL);
  mm_256_256(A, N, B, N, C, N);  // this is just the 256
  gettimeofday(&end,   NULL);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (0) printf("C[%d,%d]=%f C2=%f\n", i, j, C[i*N+j], C2[i*N+j]);
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (0) printf("C[%d,%d]=%f C2=%f\n", i, j, C[i*N+j], C2[i*N+j]);
      assert(C[i*N+j] == C2[i*N+j]);
    }
  }
  double diff = tdiff(&start, &end);
  printf("N=%4d %s basecase=%d time=%.6fs %.3f GFLOPS\n",
         N, NATIVE_AVX ? "(AVX native)" : "(AVX emulated)",
         BC, diff, 1e-9*(double)N*(double)N*(double)N*2.0/diff);

#if 0
#define REPEAT 2
  gettimeofday(&start, NULL);
  for (int i = 0; i < REPEAT; i++) {
    mm_256_256(A, B, C, N);
  }
  gettimeofday(&end,   NULL);
  printf("N=%4d R=%d time=%.6fs %.3f GFLOPS (C[0]=%f)\n",
         N, REPEAT, diff, 1e-9*(double)N*(double)N*(double)N*2.0*REPEAT/diff, C[0]);
#endif
  free(C); free(C2); free(A); free(B);
}

#define ADD_BC BC

static void mm_add_base(bool negateSecond,
                        const double *X, size_t Xstride,
                        const double *Y, size_t Ystride,
                        double *restrict Z, size_t Zstride) {
  for (size_t i = 0; i < ADD_BC; ++i) {
    for (size_t j = 0; j < ADD_BC; ++j) {
      if (negateSecond) Z[i*Zstride+j] = X[i*Xstride+j] - Y[i*Ystride+j];
      else Z[i*Zstride+j] = X[i*Xstride+j] + Y[i*Ystride+j];
    }
  }
}

static void mm_add(bool negateSecond,
                   const double *X, size_t Xstride,
                   const double *Y, size_t Ystride,
                   double *restrict Z, size_t Zstride,
                   size_t N) {
  if (N == ADD_BC) {
    mm_add_base(negateSecond, X, Xstride, Y, Ystride, Z, Zstride);
  } else {
#define S(M,r,c) (M + (r*(M ## stride) + c)*(N/2))
    _Cilk_spawn mm_add(negateSecond, S(X,0,0), Xstride, S(Y,0,0), Ystride, S(Z,0,0), Zstride, N/2);
    _Cilk_spawn mm_add(negateSecond, S(X,0,1), Xstride, S(Y,0,1), Ystride, S(Z,0,1), Zstride, N/2);
    _Cilk_spawn mm_add(negateSecond, S(X,1,0), Xstride, S(Y,1,0), Ystride, S(Z,1,0), Zstride, N/2);
    mm_add(negateSecond, S(X,1,1), Xstride, S(Y,1,1), Ystride, S(Z,1,1), Zstride, N/2);
#undef S
  }
/*   if (negateSecond) { */
/* #pragma cilk grainsize=ADD_BC */
/*     _Cilk_for(size_t i = 0; i < N; ++i) { */
/* #pragma cilk grainsize=ADD_BC */
/*       _Cilk_for(size_t j = 0; j < N; ++j) { */
/*         Z[i*Zstride + j] = X[i*Xstride + j] - Y[i*Ystride + j]; */
/*       } */
/*     } */
/*   } else { */
/* #pragma cilk grainsize=ADD_BC */
/*     _Cilk_for(size_t i = 0; i < N; ++i) { */
/* #pragma cilk grainsize=ADD_BC */
/*       _Cilk_for(size_t j = 0; j < N; ++j) { */
/*         Z[i*Zstride + j] = X[i*Xstride + j] + Y[i*Ystride + j]; */
/*       } */
/*     } */
/*   } */
}

static void mm_inc_base(bool negateSecond,
                        double *X, size_t Xstride,
                        const double *Y, size_t Ystride) {
  for (size_t i = 0; i < ADD_BC; ++i) {
    for (size_t j = 0; j < ADD_BC; ++j) {
      if (negateSecond) X[i*Xstride+j] -= Y[i*Ystride+j];
      else X[i*Xstride+j] += Y[i*Ystride+j];
    }
  }
}

static void mm_inc(bool negateSecond,
                   double *X, size_t Xstride,
                   const double *Y, size_t Ystride,
                   size_t N) {
  if (N == ADD_BC) {
    mm_inc_base(negateSecond, X, Xstride, Y, Ystride);
  } else {
#define S(M,r,c) (M + (r*(M ## stride) + c)*(N/2))
    _Cilk_spawn mm_inc(negateSecond, S(X,0,0), Xstride, S(Y,0,0), Ystride, N/2);
    _Cilk_spawn mm_inc(negateSecond, S(X,0,1), Xstride, S(Y,0,1), Ystride, N/2);
    _Cilk_spawn mm_inc(negateSecond, S(X,1,0), Xstride, S(Y,1,0), Ystride, N/2);
    mm_inc(negateSecond, S(X,1,1), Xstride, S(Y,1,1), Ystride, N/2);
#undef S
  }
/*   if (negateSecond) { */
/* #pragma cilk grainsize=ADD_BC */
/*     _Cilk_for(size_t i = 0; i < N; ++i) { */
/* #pragma cilk grainsize=ADD_BC */
/*       _Cilk_for(size_t j = 0; j < N; ++j) { */
/*         X[i*Xstride + j] -= Y[i*Ystride + j]; */
/*       } */
/*     } */
/*   } else { */
/* #pragma cilk grainsize=ADD_BC */
/*     _Cilk_for(size_t i = 0; i < N; ++i) { */
/* #pragma cilk grainsize=ADD_BC */
/*       _Cilk_for(size_t j = 0; j < N; ++j) { */
/*         X[i*Xstride + j] += Y[i*Ystride + j]; */
/*       } */
/*     } */
/*   } */
}

static void mm_addinc_base(bool negateSecond,
                           const double *X, size_t Xstride,
                           const double *Y, size_t Ystride,
                           double *restrict Z, size_t Zstride) {
  for (size_t i = 0; i < ADD_BC; ++i) {
    for (size_t j = 0; j < ADD_BC; ++j) {
      if (negateSecond) Z[i*Zstride+j] += X[i*Xstride+j] - Y[i*Ystride+j];
      else Z[i*Zstride+j] += X[i*Xstride+j] + Y[i*Ystride+j];
    }
  }
}

static void mm_addinc(bool negateSecond,
                      const double *X, size_t Xstride,
                      const double *Y, size_t Ystride,
                      double *restrict Z, size_t Zstride,
                      size_t N) {
  if (N == ADD_BC) {
    mm_addinc_base(negateSecond, X, Xstride, Y, Ystride, Z, Zstride);
  } else {
#define S(M,r,c) (M + (r*(M ## stride) + c)*(N/2))
    _Cilk_spawn mm_addinc(negateSecond, S(X,0,0), Xstride, S(Y,0,0), Ystride, S(Z,0,0), Zstride, N/2);
    _Cilk_spawn mm_addinc(negateSecond, S(X,0,1), Xstride, S(Y,0,1), Ystride, S(Z,0,1), Zstride, N/2);
    _Cilk_spawn mm_addinc(negateSecond, S(X,1,0), Xstride, S(Y,1,0), Ystride, S(Z,1,0), Zstride, N/2);
    mm_addinc(negateSecond, S(X,1,1), Xstride, S(Y,1,1), Ystride, S(Z,1,1), Zstride, N/2);
#undef S
  }
/*   if (negateSecond) { */
/* #pragma cilk grainsize=ADD_BC */
/*     _Cilk_for(size_t i = 0; i < N; ++i) { */
/* #pragma cilk grainsize=ADD_BC */
/*       _Cilk_for(size_t j = 0; j < N; ++j) { */
/*         Z[i*Zstride + j] += X[i*Xstride + j] - Y[i*Ystride + j]; */
/*       } */
/*     } */
/*   } else { */
/* #pragma cilk grainsize=ADD_BC */
/*     _Cilk_for(size_t i = 0; i < N; ++i) { */
/* #pragma cilk grainsize=ADD_BC */
/*       _Cilk_for(size_t j = 0; j < N; ++j) { */
/*         Z[i*Zstride + j] += X[i*Xstride + j] + Y[i*Ystride + j]; */
/*       } */
/*     } */
/*   } */
}

static void mm_dac(const double *A, size_t Astride,
                   const double *B, size_t Bstride,
                   double * restrict C, size_t Cstride,
                   size_t N) {
  if (N == BC) {
    //  mm_naive(A, B, C, N, N, N, stride, stride, stride);
    mm_256_256(A, Astride, B, Bstride, C, Cstride);
  } else {
#define S(M,r,c) (M + (r*(M ## stride) + c)*(N/2))
    _Cilk_spawn mm_dac(S(A,0,0), Astride, S(B,0,0), Bstride, S(C,0,0), Cstride, N/2);
    _Cilk_spawn mm_dac(S(A,0,0), Astride, S(B,0,1), Bstride, S(C,0,1), Cstride, N/2);
    _Cilk_spawn mm_dac(S(A,1,0), Astride, S(B,0,0), Bstride, S(C,1,0), Cstride, N/2);
    mm_dac(S(A,1,0), Astride, S(B,0,1), Bstride, S(C,1,1), Cstride, N/2);
    _Cilk_sync;
    _Cilk_spawn mm_dac(S(A,0,1), Astride, S(B,1,0), Bstride, S(C,0,0), Cstride, N/2);
    _Cilk_spawn mm_dac(S(A,0,1), Astride, S(B,1,1), Bstride, S(C,0,1), Cstride, N/2);
    _Cilk_spawn mm_dac(S(A,1,1), Astride, S(B,1,0), Bstride, S(C,1,0), Cstride, N/2);
    mm_dac(S(A,1,1), Astride, S(B,1,1), Bstride, S(C,1,1), Cstride, N/2);
#undef S
  }
}

static void mm_strassen(const double *A, size_t Astride,
                        const double *B, size_t Bstride,
                        double * restrict C, size_t Cstride,
                        size_t N);

void zero_fill(double *A, size_t N);

#define S(M,r,c) (M + (r*(M ## stride) + c)*(N))

static void mm_strassen_p1(const double *A, size_t Astride,
                           const double *B, size_t Bstride,
                           double * restrict M, size_t Mstride,
                           double * restrict D, // Dstride = N
                           size_t N) {
  size_t Xsize = N * N;

  double *X = D;

  _Cilk_spawn zero_fill(M, Mstride);
  _Cilk_spawn mm_add(false, S(A,0,0), Astride, S(A,1,1), Astride, X+0, N, N);
  mm_add(false, S(B,0,0), Bstride, S(B,1,1), Bstride, X+Xsize, N, N);
  _Cilk_sync;
  mm_strassen(X+0, N, X+Xsize, N, M, Mstride, N);
}

static void mm_strassen_p2(const double *A, size_t Astride,
                           const double *B, size_t Bstride,
                           double * restrict M, size_t Mstride,
                           double * restrict D, // Dstride = N
                           size_t N) {
  double *X = D;
  
  _Cilk_spawn zero_fill(M, Mstride);
  mm_add(false, S(A,1,0), Astride, S(A,1,1), Astride, X+0, N, N);
  _Cilk_sync;
  mm_strassen(X+0, N, S(B,0,0), Bstride, M, Mstride, N);
}

static void mm_strassen_p3(const double *A, size_t Astride,
                           const double *B, size_t Bstride,
                           double * restrict M, size_t Mstride,
                           double * restrict D, // Dstride = N
                           size_t N) {
  double *X = D;

  _Cilk_spawn zero_fill(M, Mstride);
  mm_add(true, S(B,0,1), Bstride, S(B,1,1), Bstride, X+0, N, N);
  _Cilk_sync;
  mm_strassen(S(A,0,0), Astride, X+0, N, M, Mstride, N);
}

static void mm_strassen_p4(const double *A, size_t Astride,
                           const double *B, size_t Bstride,
                           double * restrict M, size_t Mstride,
                           double * restrict D, // Dstride = N
                           size_t N) {
  double *X = D;
  
  _Cilk_spawn zero_fill(M, Mstride);
  mm_add(true, S(B,1,0), Bstride, S(B,0,0), Bstride, X+0, N, N);
  _Cilk_sync;
  mm_strassen(S(A,1,1), Astride, X+0, N, M, Mstride, N);
}

static void mm_strassen_p5(const double *A, size_t Astride,
                           const double *B, size_t Bstride,
                           double * restrict M, size_t Mstride,
                           double * restrict D, // Dstride = N
                           size_t N) {
  double *X = D;

  _Cilk_spawn zero_fill(M, Mstride);
  mm_add(false, S(A,0,0), Astride, S(A,0,1), Astride, X+0, N, N);
  _Cilk_sync;
  mm_strassen(X+0, N, S(B,1,1), Bstride, M, Mstride, N);
}

static void mm_strassen_p6(const double *A, size_t Astride,
                           const double *B, size_t Bstride,
                           double * restrict M, size_t Mstride,
                           double * restrict D, // Dstride = N
                           size_t N) {
  size_t Xsize = N * N;
  double *X = D;

  _Cilk_spawn mm_add(true, S(A,1,0), Astride, S(A,0,0), Astride, X+0, N, N);
  mm_add(false, S(B,0,0), Bstride, S(B,0,1), Bstride, X+Xsize, N, N);
  _Cilk_sync;
  mm_strassen(X+0, N, X+Xsize, N, M, Mstride, N);
}

static void mm_strassen_p7(const double *A, size_t Astride,
                           const double *B, size_t Bstride,
                           double * restrict M, size_t Mstride,
                           double * restrict D, // Dstride = N
                           size_t N) {
  size_t Xsize = N * N;
  double *X = D;

  _Cilk_spawn mm_add(true, S(A,0,1), Astride, S(A,1,1), Astride, X+0, N, N);
  mm_add(false, S(B,1,0), Bstride, S(B,1,1), Bstride, X+Xsize, N, N);
  _Cilk_sync;
  mm_strassen(X+0, N, X+Xsize, N, M, Mstride, N);
}

#undef S

#define DAC_BC 2048

static void mm_strassen(const double *A, size_t Astride,
                        const double *B, size_t Bstride,
                        double * restrict C, size_t Cstride,
                        size_t N) {
  /* if (N == BC) { */
  /*   mm_naive(A, B, C, N, N, N, Astride, Bstride, Cstride); */
  if (N == DAC_BC) {
    mm_dac(A, Astride, B, Bstride, C, Cstride, N);
  } else {
#define S(M,r,c) (M + (r*(M ## stride) + c)*(N/2))
#define tmp(M, i) (M + (N*(N/4)*i))

    // Goal:
    // C00 = P1 + P4 - P5 + P7
    // C01 = P3 + P5
    // C10 = P2 + P4
    // C11 = P1 + P3 - P2 + P6

    // Temporaries tmp(T,{0-3}) store intermediate products.
    // Temporaries tmp(T,{4-7}) are used as scratch space.

    /* double *T = malloc_aligned(sizeof(*A) * N * N * 2); */

    struct TmpStack *Tview = &(REDUCER_VIEW(tmp_stack));
    if (NULL == Tview->T) {
      size_t Tsize = ((N * N) - (DAC_BC * DAC_BC)) * 8 / 3;
      /* if (NULL != TS.T && */
      /*     TS.size >= Tsize) { */
      /*   fprintf(stderr, "repeat steal\n"); */
      /*   *Tview = TS; */
      /*   Tview->start = 0; */
      /*   TS.T = NULL; */
      /* } else { */
        Tview->T = malloc_aligned(sizeof(*A) * Tsize);
        Tview->start = 0;
#if DEBUG_TMP_MALLOC
        Tview->size = Tsize;
#endif
      /* } */
    } else {
      Tview->start += N * N * 4;
    }
    /* fprintf(stderr, "N = %lu, Tview->T = %p, Tview->size = %lu, Tview->start = %lu\n", N, Tview->T, Tview->size, Tview->start); */
#if DEBUG_TMP_MALLOC
    assert(Tview->size >= Tview->start + (N * N * 2));
#endif
    double *T = Tview->T + Tview->start;

    // T0 = P2
    _Cilk_spawn mm_strassen_p2(A, Astride, B, Bstride,
                               tmp(T,0), N/2, tmp(T,4), N/2);
    /* printf("P2:\n"); */
    /* print_matrix(tmp(T,0), N/2, N/2, N/2, true); */
    // T1 = P3
    _Cilk_spawn mm_strassen_p3(A, Astride, B, Bstride,
                               tmp(T,1), N/2, tmp(T,5), N/2);
    /* printf("P3:\n"); */
    /* print_matrix(tmp(T,1), N/2, N/2, N/2, true); */
    // T2 = P4
    _Cilk_spawn mm_strassen_p4(A, Astride, B, Bstride,
                               tmp(T,2), N/2, tmp(T,6), N/2);
    /* printf("P4:\n"); */
    /* print_matrix(tmp(T,2), N/2, N/2, N/2, true); */
    // T3 = P5
    _Cilk_spawn mm_strassen_p5(A, Astride, B, Bstride,
                               tmp(T,3), N/2, tmp(T,7), N/2);
    _Cilk_sync;
    /* printf("P5:\n"); */
    /* print_matrix(tmp(T,3), N/2, N/2, N/2, true); */

    // C00 = P4 - P5
    _Cilk_spawn mm_addinc(true, tmp(T,2), N/2, tmp(T,3), N/2,
                          S(C,0,0), Cstride, N/2);
    // C01 = P3 + P5
    _Cilk_spawn mm_addinc(false, tmp(T,1), N/2, tmp(T,3), N/2,
                          S(C,0,1), Cstride, N/2);
    // C10 = P2 + P4
    _Cilk_spawn mm_addinc(false, tmp(T,0), N/2, tmp(T,2), N/2,
                          S(C,1,0), Cstride, N/2);
    // C11 = P3 - P2
    mm_addinc(true, tmp(T,1), N/2, tmp(T,0), N/2,
              S(C,1,1), Cstride, N/2);
    _Cilk_sync;
    /* printf("C00:\n"); */
    /* print_matrix(S(C,0,0), N/2, N/2, Cstride, true); */
    /* printf("C11:\n"); */
    /* print_matrix(S(C,1,1), N/2, N/2, Cstride, true); */

    // Temporary tmp(T,0) stores P1, while temporaries tmp(T,{1-6})
    // are scratch space.  Each helper call uses two consecutive
    // scratch-space temporaries.

    // T0 = P1
    _Cilk_spawn mm_strassen_p1(A, Astride, B, Bstride,
                               tmp(T,0), N/2, tmp(T,1), N/2);
    /* printf("P1:\n"); */
    /* print_matrix(tmp(T,0), N/2, N/2, N/2, true); */
    // C00 = (P4 - P5) + P7
    _Cilk_spawn mm_strassen_p7(A, Astride, B, Bstride,
                               S(C,0,0), Cstride, tmp(T,3), N/2);
    /* printf("C00:\n"); */
    /* print_matrix(S(C,0,0), N/2, N/2, Cstride, true); */
    // C11 = (P3 - P2) + P6
    mm_strassen_p6(A, Astride, B, Bstride,
                   S(C,1,1), Cstride, tmp(T,5), N/2);
    _Cilk_sync;
    /* printf("C11:\n"); */
    /* print_matrix(S(C,1,1), N/2, N/2, Cstride, true); */

    // C00 = (P4 - P5 + P7) + P1
    _Cilk_spawn mm_inc(false, S(C,0,0), Cstride, tmp(T,0), N/2, N/2);
    // C11 = (P3 - P2 + P6) + P1
    mm_inc(false, S(C,1,1), Cstride, tmp(T,0), N/2, N/2);
    _Cilk_sync;

    /* free(T); */
    Tview->start -= N * N * 4;
#undef S
  }
}

void mm(double *A, double *B, double *C, size_t N) {
  // requires N is a multiple of BC  (Probably doesn't really require that.)
  if (0) {
    for (size_t i = 0; i < N; i+=BC) {
      for (size_t k = 0; k < N; k+=BC) {
        for (size_t j = 0; j < N; j+=BC) {
          printf("%8ld %8ld %8ld\n", i*N+j, j*N+k, i*N+k);
          mm_256_256(A+i*N+j, N, B+j*N+k, N, C+i*N+k, N);
        }
      }
    }
  } else {
    CILK_C_REGISTER_REDUCER(tmp_stack);
    mm_strassen(A, N, B, N, C, N, N);
    CILK_C_UNREGISTER_REDUCER(tmp_stack);
  }
}

void random_fill(double *A, size_t N) {
  for (size_t i = 0; i < N*N; ++i) {
    A[i] = (double)rand() / (double)RAND_MAX;
  }
}
void nonrandom_fill(double *A, size_t N) {
  for (size_t i = 0; i < N*N; ++i) {
    A[i] = i;
  }
  /* for (size_t i = 0; i < N; ++i) { */
  /*   for (size_t j = 0; j < N; ++j) { */
  /*     /\* if (i == N - j - 1 || *\/ */
  /*     /\*     i == j) *\/ */
  /*     if (i == j) */
  /*       A[i*N + j] = 1; */
  /*     else */
  /*       A[i*N + j] = 0; */
  /*   } */
  /* } */
}

void zero_fill(double *A, size_t N) {
  #pragma cilk grainsize=BC*BC
  _Cilk_for (size_t i = 0; i < N*N; ++i) {
  /* for (size_t i = 0; i < N*N; ++i) { */
    A[i] = 0;
  }
}


#ifndef n
#define n 8192
#endif

#define FINAL_VERIFY 0
void test3(void) {
  double *C  = malloc(n*n*sizeof(*C));
  double *C2 = malloc(n*n*sizeof(*C2));
  double *A  = malloc(n*n*sizeof(*A));
  double *B  = malloc(n*n*sizeof(*B));
  if (FINAL_VERIFY) {
    nonrandom_fill(A, n);
    nonrandom_fill(B, n);
  } else {
    random_fill(A, n);
    random_fill(B, n);
  }
  zero_fill(C, n);
  zero_fill(C2, n);

  /* printf("A:\n");  print_matrix(A, n, n, n, true); */
  /* printf("B:\n");  print_matrix(B, n, n, n, true); */

  struct timeval start, end;
  gettimeofday(&start, NULL);
  mm(A, B, C, n);
  gettimeofday(&end, NULL);
  double diff = tdiff(&start, &end);
  // printf("N=%4d time=%.6fs %.3f GFLOPS\n", n, diff, 1e-9*(double)n*(double)n*(double)n*2.0/diff);
  printf("%.6f\n", diff);

  /* printf("fast:\n");  print_matrix(C, n, n, n, true); */

  if (FINAL_VERIFY) {
    mm_naive(A, B, C2, n, n, n, n, n, n);
    /* printf("naive:\n"); print_matrix(C2, n, n, n, true); */
    for (size_t i = 0; i < n*n; i++) {
      abort_if_too_different(C[i], C2[i], i);
    }
  }
  free(C); free(C2); free(A); free(B);
}

int main(int argc __attribute__((__unused__)), char *argv[] __attribute__((__unused__))) {
  // test1();
  // test2(BC);
  test3();
  return 0;
}
