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

static float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) +1e-6*(end->tv_usec-start->tv_usec);
}

#include "./verify.h"


///<<----------------------------------------------------------------------
void mmdac(double *restrict C, const double *A, const double *B, int size) {
  if (size <= 1) {
    *C += *A * *B;
  } else {
    int s00 = 0;                                     ///\lilabel{s00}
    int s01 = size/2;                                ///\lilabel{s01}
    int s10 = (size/2)*n;                            ///\lilabel{s10}
    int s11 = (size/2)*(n+1);                        ///\lilabel{s11}
    cilk_spawn mmdac(C+s00, A+s00, B+s00, size/2);   ///\lilabel{A00B00}
    cilk_spawn mmdac(C+s01, A+s00, B+s01, size/2);   ///\lilabel{A00B01}
    cilk_spawn mmdac(C+s10, A+s10, B+s00, size/2);   ///\lilabel{A10B00}
               mmdac(C+s11, A+s10, B+s01, size/2);   ///\lilabel{A10B01}
    cilk_sync;                                       ///\lilabel{sync1}
    cilk_spawn mmdac(C+s00, A+s01, B+s10, size/2);   ///\lilabel{A01B10}
    cilk_spawn mmdac(C+s01, A+s01, B+s11, size/2);   ///\lilabel{A01B11}
    cilk_spawn mmdac(C+s10, A+s11, B+s10, size/2);   ///\lilabel{A11B10}
               mmdac(C+s11, A+s11, B+s11, size/2);   ///\lilabel{A11B11}
    cilk_sync;                                       ///\lilabel{sync2}
  }
}
///>>----------------------------------------------------------------------
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

