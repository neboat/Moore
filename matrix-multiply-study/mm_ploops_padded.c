// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cilk/cilk.h>

/* Allow us to change n on the compiler command line with for example -Dn=1024 */
#ifndef n
#define n 4096
#endif
#define PAD 2
double A[n][n+PAD];
double B[n][n+PAD];
double C[n][n+PAD];

static float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) +1e-6*(end->tv_usec-start->tv_usec);
}

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
  cilk_for (int i = 0; i < n; ++i) {                    ///\lilabel{loop_i}
    cilk_for (int j = 0; j < n; ++j) {                  ///\lilabel{loop_j}
      for (int k = 0; k < n; ++k) {                     ///\lilabel{loop_k}
        C[i][j] += A[i][k] * B[k][j];                   ///\lilabel{multiply}
      }
    }
  }                                                 ///\lilabel{loops_t}
///>>----------------------------------------------------------------------
  gettimeofday(&end, NULL);
  printf("%0.6f\n", tdiff(&start, &end));
  if (need_to_verify) verify();
  return 0;
}

