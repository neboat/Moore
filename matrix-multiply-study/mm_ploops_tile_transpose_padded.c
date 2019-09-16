// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cilk/cilk.h>

/* Allow us to change n on the compiler command line with for example -Dn=1024 */
#ifndef n
#define n 4096
#endif
#define PAD 6
double A[n][n+PAD];
double B[n][n+PAD];
double Bt[n][n+PAD];
double C[n][n+PAD];

static float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec)
    +1e-6*(end->tv_usec-start->tv_usec);
}

#include "./verify.h"

#define tilesize 128

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
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Bt[i][j] = B[j][i];
    }
  }

  cilk_for (int ih = 0; ih < n; ih+=tilesize) {                     ///\lilabel{block_loop_i}
    cilk_for (int kh = 0; kh < n; kh+=tilesize) {                   ///\lilabel{block_loop_k}
      for (int jh = 0; jh < n; jh+=tilesize) {                      ///\lilabel{block_loop_j}
        for (int il = 0; il < tilesize; il++) {                     ///\lilabel{base_loop_i}
          for (int kl = 0; kl < tilesize; kl++) {                   ///\lilabel{base_loop_j}
            for (int jl = 0; jl < tilesize; ++jl) {                 ///\lilabel{base_loop_k}
              C[ih+il][kh+kl] += A[ih+il][jh+jl]*Bt[kh+kl][jh+jl];  ///\lilabel{base_multiply}
            }
          }
        }
      }
    }
  }
  ///>>----------------------------------------------------------------------
  gettimeofday(&end, NULL);
  printf("%0.6f\n", tdiff(&start, &end));
  if (need_to_verify) verify();
  return 0;
}
/* Local Variables:      */
/* mode: C               */
/* c-basic-offset:  2    */
/* indent-tabs-mode: nil */
/* End:                  */

