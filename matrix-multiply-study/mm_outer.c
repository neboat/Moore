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
// M is typically 256, it's the size of the small matrix that will operate on.
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

static inline void mm_256_256(
    const double *A, /* \cvrb|A[BC][BC]| row-major order with stride of \cvrb|N|. */
    const double *B, /* \cvrb|B[BC][BC]| row-major order with stride of \cvrb|N|. */
    /*out*/ double *restrict C, /* \cvrb|C[BC][BC]| row-major order with stride of \cvrb|N|. */
    size_t N) {
  // Effect: Compute $\cvrb|C|=\cvrb|C|+\cvrb|A|\cdot\cvrb|B|$
  //  where \cvrb|A|, \cvrb|B|, and \cvrb|C| are $\cvrb|BC|\times\cvrb|BC|$ matrices
  //  with stride \cvrb|N|.
  //  In this case BC is 256 and N is 4096.

  // \cvrb|AT| is a temporary $\cvrb|BC|\times\cvrb|BC|$ array.
  // \cvrb|AT| is blocked with address calculations.
  if (AT == NULL) AT = malloc_aligned(sizeof(*A) * BC * BC);  ///\lilabel{createAT}
  if (BT == NULL) BT = malloc_aligned(sizeof(*B) * BC * BC);  ///\lilabel{createBT}
  for (size_t i = 0; i < BC; i++) {         ///\lilabel{blockAB(}
    for (size_t j = 0; j < BC; j++) {
      *(A_(AT, i, j)) = A[i * N + j];
      *(B_(BT, i, j)) = B[i * N + j];
    }
  }                                     ///\lilabel{blockAB)}
  for (size_t i = 0; i < BC; i += 8) {      ///\lilabel{computeC(}
    for (size_t k = 0; k < BC; k += 4) {
      ///>>----------------------------------------------------------------------
      // printf("Mult AT[%d,*] by BT[*,%d] into C[%d][%d]\n", i, k, i, k);
      ///<<----------------------------------------------------------------------
      mm_8_4_256_256(A_(AT, i, 0), B_(BT, 0, k), C + i * N + k, N);
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
  mm_256_256(A, B, C, N);  // this is just the 256
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

static void mm_dac(const double *A, const double *B, double * __restrict__ C,
                   size_t N, size_t stride) {
  if (N == BC) {
    //  mm_naive(A, B, C, N, N, N, stride, stride, stride);
    mm_256_256(A, B, C, stride);
  } else {
    size_t s0 = 0;
    size_t s1 = N/2;
    size_t s2 = (N/2)*stride;
    size_t s3 = (N/2)*(stride+1);
    _Cilk_spawn mm_dac(A+s0, B+s0, C+s0, N/2, stride);  // printf("%8ld %8ld %8ld\n", s0, s0, s0);
    _Cilk_spawn mm_dac(A+s0, B+s1, C+s1, N/2, stride);  // printf("%8ld %8ld %8ld\n", s0, s1, s1);

    _Cilk_spawn mm_dac(A+s2, B+s0, C+s2, N/2, stride);  // printf("%8ld %8ld %8ld\n", s2, s0, s2);
    mm_dac(A+s2, B+s1, C+s3, N/2, stride);  // printf("%8ld %8ld %8ld\n", s2, s1, s3);
    _Cilk_sync;

    _Cilk_spawn mm_dac(A+s1, B+s2, C+s0, N/2, stride);  // printf("%8ld %8ld %8ld\n", s1, s2, s0);
    _Cilk_spawn mm_dac(A+s1, B+s3, C+s1, N/2, stride);  // printf("%8ld %8ld %8ld\n", s1, s3, s1);

    _Cilk_spawn mm_dac(A+s3, B+s2, C+s2, N/2, stride);  // printf("%8ld %8ld %8ld\n", s3, s2, s2);
    mm_dac(A+s3, B+s3, C+s3, N/2, stride);  // printf("%8ld %8ld %8ld\n", s3, s3, s3);
  }
}

void mm(double *A, double *B, double *C, size_t N) {
  // requires N is a multiple of BC  (Probably doesn't really require that.)
  if (0) {
    for (size_t i = 0; i < N; i+=BC) {
      for (size_t k = 0; k < N; k+=BC) {
        for (size_t j = 0; j < N; j+=BC) {
          printf("%8ld %8ld %8ld\n", i*N+j, j*N+k, i*N+k);
          mm_256_256(A+i*N+j, B+j*N+k, C+i*N+k, N);
        }
      }
    }
  } else {
    mm_dac(A, B, C, N, N);
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
}

void zero_fill(double *A, size_t N) {
  for (size_t i = 0; i < N*N; ++i) {
    A[i] = 0;
  }
}


#define FINAL_VERIFY 0
#ifndef n
#define n 4096
#endif
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

  struct timeval start, end;
  gettimeofday(&start, NULL);
  mm(A, B, C, n);
  gettimeofday(&end, NULL);
  double diff = tdiff(&start, &end);
  // printf("N=%4d time=%.6fs %.3f GFLOPS\n", n, diff, 1e-9*(double)n*(double)n*(double)n*2.0/diff);
  printf("%.6f\n", diff);

  // printf("fast:\n");  print_matrix(C, n/2, n/2, n, true);

  if (FINAL_VERIFY) {
    mm_naive(A, B, C2, n, n, n, n, n, n);
    // printf("naive:\n"); print_matrix(C2, N/2, N/2, N, true);
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
