// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#ifndef n
#define n 4096
#endif
double A[n][n];                                  ///\lilabel{matrix_decl(}
double B[n][n];
double C[n][n];                                  ///\lilabel{matrix_decl)}

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

int main(int argc __attribute__((unused)),
         const char *argv[] __attribute__((unused))) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i][j] = (double)rand() / (double)RAND_MAX;
      B[i][j] = (double)rand() / (double)RAND_MAX;
      C[i][j] = 0;
    }
  }

  struct timeval start, end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < n; ++i) {                    ///\lilabel{loop_i} \lilabel{loop_nest(}
    for (int j = 0; j < n; ++j) {                  ///\lilabel{loop_j}
      for (int k = 0; k < n; ++k) {                ///\lilabel{loop_k}\lilabel{loop_k(}
        C[i][j] += A[i][k] * B[k][j];              ///\lilabel{multiply}
      }                                            ///\lilabel{loop_k)}
    }
  }                                                ///\lilabel{loop_nest)}

  gettimeofday(&end, NULL);
  printf("%0.6f\n", tdiff(&start, &end));
  return 0;
}
