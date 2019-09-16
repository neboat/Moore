// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

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


#ifndef THRESH
///<<----------------------------------------------------------------------
#line 1
#define THRESH 32                                ///\lilabel{THRESH}
///>>----------------------------------------------------------------------
#endif
///<<----------------------------------------------------------------------
#line 2

void mmbase(double *restrict C, double *restrict A, double *restrict B) {  ///\lilabel{mmbase(}
  double Ac[THRESH*THRESH], Bc[THRESH*THRESH];               ///\lilabel{allocate_Ac_Bc}
  for (size_t i = 0; i < THRESH; i++) {                      ///\lilabel{transpose_loop_i}\lilabel{transpose(}
    for (size_t j = 0; j < THRESH; j++) {                    ///\lilabel{transpose_loop_j}
      Ac[i*THRESH+j] = A[i*n+j];  // copy                    ///\lilabel{copy_A}
      Bc[j*THRESH+i] = B[i*n+j];  // copy and transpose      ///\lilabel{copy_B}
    }
  }                                                          ///\lilabel{transpose)}

  for (size_t i = 0; i < THRESH; ++i) {                      ///\lilabel{loop_i} \lilabel{loop_nest(}
    for (size_t j = 0; j < THRESH; ++j) {                    ///\lilabel{loop_j}
      for (size_t k = 0; k < THRESH; ++k) {                  ///\lilabel{loop_k}\lilabel{loop_k(}
        C[i*n+j] += Ac[i*THRESH+k] * Bc[j*THRESH+k];         ///\lilabel{multiply}
      }                                                      ///\lilabel{loop_k)}
    }
  }                                                          ///\lilabel{loop_nest)}
}                                                            ///\lilabel{mmbase)}

void mmdac(double *restrict C, double *restrict A, double *restrict B,
           size_t size) {
  if (size == THRESH) {                                      ///\lilabel{base_case_check}
    mmbase(C, A, B);                                         ///\lilabel{call_base_case}
  } else {
    // ...                                                   ///\lilabel{recur}
///>>----------------------------------------------------------------------
    size_t s0 = 0;
    size_t s1 = size/2;
    size_t s2 = (size/2) * n;
    size_t s3 = (size/2) * (n+1);
    mmdac(C+s0, A+s0, B+s0, size/2);
    mmdac(C+s1, A+s0, B+s1, size/2);
    mmdac(C+s2, A+s2, B+s0, size/2);
    mmdac(C+s3, A+s2, B+s1, size/2);
    mmdac(C+s0, A+s1, B+s2, size/2);
    mmdac(C+s1, A+s1, B+s3, size/2);
    mmdac(C+s2, A+s3, B+s2, size/2);
    mmdac(C+s3, A+s3, B+s3, size/2);
///<<----------------------------------------------------------------------
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
