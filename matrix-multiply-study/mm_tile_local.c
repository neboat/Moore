// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
/** BEGIN HIDDEN **/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/* Allow us to change n on the compiler command line with for example -Dn=1024 */
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

#define tilesize 32

void mmbase(double *restrict C, double *restrict A, double *restrict B) {
  double Ac[tilesize*tilesize], Bc[tilesize*tilesize];
  for (int i = 0; i < tilesize; i++) {
    for (int j = 0; j < tilesize; j++) {
      Ac[i*tilesize+j]=A[i*n+j];
      Bc[i*tilesize+j]=B[i*n+j];  // DON'T transpose it (for SCIENCE!)
    }
  }
  for (int i = 0; i < tilesize; ++i) {
    for (int j = 0; j < tilesize; ++j) {
      for (int k = 0; k < tilesize; ++k) {
        C[i*n+k] += Ac[i*tilesize+j]*Bc[j*tilesize+k];
      }
    }
  }
}

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
  /** END HIDDEN **/
  for (int ih = 0; ih < n; ih += tilesize) {                   ///\lilabel{block_loop_i}
    for (int jh = 0; jh < n; jh += tilesize) {                 ///\lilabel{block_loop_k}
      for (int kh = 0; kh < n; kh += tilesize) {                      ///\lilabel{block_loop_j}
        // for (int il = 0; il < tilesize; il++) {                       ///\lilabel{base_loop_i}
        //   for (int kl = 0; kl < tilesize; ++kl) {               ///\lilabel{base_loop_k}
        //     for (int jl = 0; jl < tilesize; jl++) {                     ///\lilabel{base_loop_j}
        //       C[ih+il][jh+jl] += A[ih+il][kh+kl] * B[kh+kl][jh+jl]; ///\lilabel{base_multiply}
        // } } }
        mmbase(&C[ih][jh], &A[ih][kh], &B[kh][jh]);
  } } }
  /** BEGIN HIDDEN **/
  gettimeofday(&end, NULL);
  printf("%0.6f\n", tdiff(&start, &end));
  if (need_to_verify) verify();
  return 0;
}
/** END HIDDEN **/
