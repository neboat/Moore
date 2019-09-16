// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
struct my4 { double t[4] __attribute__((aligned(64))); };

static inline void addstore4(__m256d v,
                             double *a,
                             double *b,
                             double *c,
                             double *d) {
  // Effect: Add the elements of \cvrb|v| into \cvrb|*a|, \cvrb|*b|,
  // \cvrb|*c|, and \cvrb|*d| respectively.
  struct my4 m;
  *((__m256d*)&m)=v;
  *a += m.t[0];
  *b += m.t[1];
  *c += m.t[2];
  *d += m.t[3];
}
