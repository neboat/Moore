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
  return (end->tv_sec-start->tv_sec)
    +1e-6*(end->tv_usec-start->tv_usec);
}


///<<----------------------------------------------------------------------
void mmbase(double *restrict C, double *restrict A, double *restrict B, int size) {
  for (int i = 0; i < size; ++i) {              ///\lilabel{loop_i}
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        C[i*n+k] += A[i*n+j]*B[j*n+k];          ///\lilabel{loops_inner}
      }
    }
  }
}

static const int DAC_THRESHOLD = 256;  ///\lilabel{DAC_THRESHOLD}

void mmdac(double *C, double *A, double *B, int size) {
  if (size <= DAC_THRESHOLD) {                        ///\lilabel{base_case_check}
    mmbase(C, A, B, size);
  } else {
    int s00 = 0;
    int s01 = size/2;
    int s10 = (size/2)*n;
    int s11 = (size/2)*(n+1);
    cilk_spawn mmdac(C+s00, A+s00, B+s00, size/2);  ///\lilabel{A00B00}
    cilk_spawn mmdac(C+s01, A+s00, B+s01, size/2);
    cilk_spawn mmdac(C+s10, A+s10, B+s00, size/2);
               mmdac(C+s11, A+s10, B+s01, size/2);
    cilk_sync;
    cilk_spawn mmdac(C+s00, A+s01, B+s10, size/2);
    cilk_spawn mmdac(C+s01, A+s01, B+s11, size/2);
    cilk_spawn mmdac(C+s10, A+s11, B+s10, size/2);
               mmdac(C+s11, A+s11, B+s11, size/2);  ///\lilabel{A11B11}
    cilk_sync;
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

