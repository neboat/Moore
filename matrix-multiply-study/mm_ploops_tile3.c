// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
/** BEGIN HIDDEN **/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cilk/cilk.h>

/* Allow us to change n on the compiler command line with for example -Dn=1024 */
#ifndef n
#define n 4096
#endif
double A[n][n] __attribute__((__align__(16)));
double B[n][n] __attribute__((__align__(16)));
double C[n][n] __attribute__((__align__(16)));

static float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) +1e-6*(end->tv_usec-start->tv_usec);
}

#include "./verify.h"

#ifndef tilesize
#define tilesize 128
#endif

void mm_tile_base(double *__restrict__ CC_, double * __restrict__ AA, double * __restrict__ BB) {
// Effect: CC, AA, and BB point to arrays that are tilesize*tilesize in size with stride n.
//    Do CC+=AA*BB;
  double *CC __attribute__((__align__(16))) = CC_;
  for (int i = 0; i < tilesize; i++) {
    for (int j = 0; j < tilesize; j++) {
      for (int k = 0; k < tilesize; k++) {
        CC[i*n+j] += AA[i*n+k] * BB[k*n + j];
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
        mm_tile_base(&C[ih][jh], &A[ih][kh], &B[kh][jh]);
  } } }
  /** BEGIN HIDDEN **/
  gettimeofday(&end, NULL);
  printf("%0.6f\n", tdiff(&start, &end));
  if (need_to_verify) verify();
  return 0;
}
/** END HIDDEN **/
