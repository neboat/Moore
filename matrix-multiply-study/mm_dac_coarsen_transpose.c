// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include <cilk/cilk.h>

#ifndef n
#define n 4096
#endif
double A[n][n];
double B[n][n];
double C[n][n];

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

#ifndef THRESH
#define THRESH 32
#endif

void mmbase(double *restrict C, double *restrict A, double *restrict B) {
#ifndef NOTRANSPOSE
  double Ac[THRESH*THRESH], Bc[THRESH*THRESH];
  for (size_t i = 0; i < THRESH; i++) {
    for (size_t j = 0; j < THRESH; j++) {
      Ac[i*THRESH+j]=A[i*n+j];  // copy
      Bc[j*THRESH+i]=B[i*n+j];  // copy and transpose
    }
  }
#endif
#ifdef NOTRANSPOSE
  for (size_t i = 0; i < THRESH; ++i) {
    for (size_t j = 0; j < THRESH; ++j) {
      for (size_t k = 0; k < THRESH; ++k) {
        C[i*n+j] += A[i*n+k] * B[k*n+j];
      }
    }
  }
#else
  for (size_t i = 0; i < THRESH; ++i) {
    for (size_t j = 0; j < THRESH; ++j) {
      for (size_t k = 0; k < THRESH; ++k) {
        C[i*n+j] += Ac[i*THRESH+k] * Bc[j*THRESH+k];
      }
    }
  }
#endif
}

///<<----------------------------------------------------------------------
void mmdac(double *restrict C, double *restrict A, double *restrict B,
           size_t size) {
  if (size == THRESH) {                             ///\lilabel{base_case_check}
    mmbase(C, A, B);                                ///\lilabel{call_base_case}
  } else {
    size_t s00 = 0;                                 ///\lilabel{s00}
    size_t s01 = size/2;                            ///\lilabel{s01}
    size_t s10 = (size/2)*n;                        ///\lilabel{s10}
    size_t s11 = (size/2)*(n+1);                    ///\lilabel{s11}
    cilk_spawn mmdac(C+s00, A+s00, B+s00, size/2);  ///\lilabel{A00B00}
    cilk_spawn mmdac(C+s01, A+s00, B+s01, size/2);  ///\lilabel{A00B01}
    cilk_spawn mmdac(C+s10, A+s10, B+s00, size/2);  ///\lilabel{A10B00}
               mmdac(C+s11, A+s10, B+s01, size/2);  ///\lilabel{A10B01}
    cilk_sync;                                      ///\lilabel{sync1}
    cilk_spawn mmdac(C+s00, A+s01, B+s10, size/2);  ///\lilabel{A01B10}
    cilk_spawn mmdac(C+s01, A+s01, B+s11, size/2);  ///\lilabel{A01B11}
    cilk_spawn mmdac(C+s10, A+s11, B+s10, size/2);  ///\lilabel{A11B10}
               mmdac(C+s11, A+s11, B+s11, size/2);  ///\lilabel{A11B11}
    cilk_sync;                                      ///\lilabel{sync2}
  }
}
///>>----------------------------------------------------------------------

#include "./verify.h"

int main(int argc, const char *argv[]) {
  parse_args(argc, argv);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i][j] = (double)rand() / (double)RAND_MAX;
      B[i][j] = (double)rand() / (double)RAND_MAX;
      C[i][j] = 0;
  } }

  struct timeval start, end;
  gettimeofday(&start, NULL);

  mmdac(&C[0][0], &A[0][0], &B[0][0], n);

  gettimeofday(&end, NULL);
  printf("%0.6f\n", tdiff(&start, &end));
  if (need_to_verify) verify();
  return 0;
}
/* Local Variables:      */
/* mode: C               */
/* End:                  */
/** END HIDDEN **/
