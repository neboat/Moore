/* -*- mode: C; c-basic-offset: 4; -*- */
// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <stdio.h>
///<<--------------------
#include <mkl_cblas.h>
// . . .
///>>--------------------
#include <stdlib.h>
// #include <mkl_blas.h>

#ifndef n
#define n 4096
#endif

float tdiff(struct timeval *start, struct timeval *end) {
    return (end->tv_sec-start->tv_sec) +1e-6*(end->tv_usec - start->tv_usec);
}

double A[n][n];
double B[n][n];
double C[n][n];

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
  ///<<----------------------------------------------------------------------
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,   ///\lilabel{call(}
              n, n, n,
              1,
              (const double *)A, n,
              (const double *)B, n,
              0,
              (double *)C, n);                             ///\lilabel{call)}
  ///>>----------------------------------------------------------------------
  gettimeofday(&end, NULL);
  //  printf("A= (%12.5g %12.5g)   B=(%12.5g %12.5g)   C=(%12.5g %12.5g)\n",
  //         A[0][0], A[0][1], B[0][0], B[0][1], C[0][0], C[0][1]);
  //  printf("   (%12.5g %12.5g)     (%12.5g %12.5g)     (%12.5g %12.5g)\n",
  //         A[1][0], A[1][1], B[1][0], B[1][1], C[1][0], C[1][1]);
  double diff = tdiff(&start, &end);
  //  printf("time=%.6fs %.3f GFLOPS\n", diff, 1e-9*(double)N*(double)N*(double)N*2.0/diff);
  printf("%.6f\n", diff);
  if (need_to_verify) verify();
  return 0;
}

