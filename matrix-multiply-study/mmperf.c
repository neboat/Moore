// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
/* Create mmperf.tex */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

struct tinfo {
  const char *machine;
  const char *timestamp;
  const char *implementation;
  const char *md5;
  double runtime;
} tinfos[] = {
#include "./timedata.c"
  {0, 0, 0, 0, 0}
};

static double find_time(const char *implementation) {
  int count = 0;
  double best = 1e30;
  for (int i = 0; tinfos[i].machine; i++) {
    if ((   (strcmp(tinfos[i].machine, "alf") == 0)
         || (strcmp(tinfos[i].machine, "temp") == 0))
        && (strcmp(tinfos[i].implementation, implementation) == 0)) {
      count++;
      double thist = tinfos[i].runtime;
      if (thist < best) best=thist;
    }
  }
  if (count == 0) return -1;
  else          return best;
}

/*
static void printnum(double v, int n_digits_after_dot) {
  char str[100];
  snprintf(str, sizeof(str), "%.*f", n_digits_after_dot, v);
  char *dot = strchr(str, '.');
  *dot = '&';
  printf("%9s", str);
}
*/

struct rowinfo {
  bool  print;
  char *key;
  char *human;
};

double last_speed=-1;
static void do_row(struct rowinfo *ri) {
  if (!ri->print)
    printf("%%");
  else
    printf(" ");
  printf("\\mmref{%s}%*s & %40s ", ri->key, (int)(30-strlen(ri->key)), "", ri->human);
  double besttime = find_time(ri->key);
  printf(" & %.2f", besttime);
  //  printnum(besttime, 2);
  double gflops = 2.0*4096.0*4096.0*4096.0/1e9/besttime;
  printf(" & %.3f ", gflops);
  //  printnum(gflops, 3);
  double pyth = find_time("mm_python");
  int absspeed = pyth/besttime + 0.5;
  printf(" & %d ", absspeed);
  //  printnum(absspeed, 1);
  //  printf(" & ");
  if (last_speed > 0) {
    //    printnum(last_speed/besttime, 1);
    printf(" & %.2f ", last_speed/besttime);
  } else {
    printf(" & ");
  }
  double peak_gflops = 2.4*8*16;
  printf(" & %2.3f", 100*gflops/peak_gflops);
  //  printnum(100*gflops/peak_gflops, 1);
  //  printf("\\%%");
  printf("\\\\\n");
  if (ri->print)
    last_speed = besttime;
}

static void do_rows(void) {
  struct rowinfo names[]={
    {true,  "mm_python", "Python"},
    {true,  "mm_java",   "Java"},
    {true,  "mm_c_gcc_O0",                    "C"},
    {true,  "mm_c_gcc_O3",                    "+ switches"},
    {false, "mm_c_icc_O3",                    "C, using ICC + switches"},
    {false, "mm_permute",                     "C + permute loops"},
    {true,  "mm_transpose",                   "+ transpose"},
    {true,  "mm_sdac",                        "Divide-and-conquer"},
    {true,  "mm_sdac_coarsen_transpose",      "+ coarsening, transpose"},
    {true,  "mm_sdac_coarsen_transpose_vec",  "+ vectorization"},
    {false, "mm_ploops_2P",                   "Parallel loops $P=2$"},
    {false, "mm_ploops",                      "Parallel loops"},
    {false, "mm_ploops_tile",                 "+ tiling"},
    {false, "mm_ploops_tile_transpose",       "+ transpose"},
    {false, "mm_dac",                         "Parallel divide-and-conquer"},
    {false, "mm_dac_coarsen",                 "+ coarsening"},
    {true,  "mm_dac_coarsen_transpose",       "Parallel divide-and-conquer"},
    // "+ transpose"}, includes vectorization
    {false, "mm_dac_coarsen_transpose_axAVX", "+ machine-specific compilation"},
    {true,  "mm_dac_coarsen_transpose_gccmavx", "+ machine-specific compilation"},
    {true,  "mm_outer",                       "+ AVX intrinsics"},
    {true,  "mm_mkl_blas",                    "Intel MKL"},
    {false, "mm_mkl_blas_shared",             "Intel MKL with 1-core background job"},
    {false, 0, 0}};
  for (int i = 0; names[i].key; i++) {
    do_row(&names[i]);
  }
}

int main(int argc __attribute__((__unused__)), char *argv[]  __attribute__((__unused__))) {
  printf("%% -*- mode: latex; tex-fontify-script: nil; -*-\n");
  printf("%% Autogenerated by mmperf.c\n");
  // printf("\\begin{tabular}[t]{|d{2.0}ld{5.2}d{3.3}d{5.1}d{2.1}d{2.3}|}%%\n");
  // printf("\\hline\n");
  printf("\\begin{tab}{d{2.0}ld{5.2}d{3.3}d{5.1}d{2.1}d{2.3}}%%\n");
  printf("\\toprule\n");
  {
    char *names[][2]={{"Running", "time"},
                      {"", "GFLOPS"},
                      {"Absolute", "speedup"},
                      {"Relative", "speedup"},
                      {"Percent",  "of peak"},
                      {0, 0}};
    int i;
    printf("& ");
    for (i = 0; names[i+1][0]; i++) {
      printf(" &\\multicolumn{1}{c}{\\textit{%s}}\n", names[i][0]);
    }
    printf(" &\\multicolumn{1}{c}{\\textit{%s}}\n", names[i][0]);
    printf("\\\\[-2pt]");
    printf("\\multicolumn{1}{c}{\\textit{Version}}\n");
    printf(" & \\multicolumn{1}{l}{\\textit{Implementation}}\n");
    for (int i = 0; names[i+1][0]; i++) {
      printf(" &\\multicolumn{1}{c}{\\textit{%s}}\n", names[i][1]);
    }
    printf(" &\\multicolumn{1}{c}{\\textit{%s}}\n", names[i][1]);
    printf("\\\\\n");
    // printf("\\hline\n");
    printf("\\midrule\n");
  }
  do_rows();
  // printf("\\hline\n");
  printf("\\bottomrule\n");
  printf("\\end{tab}%%\n");

  struct save_perf_var {
    const char *key;
    const char *macroname;
  } save_these[] = {{"mm_c_gcc_O3", "mmcgccothree"}};
  for (size_t i = 0; i < sizeof(save_these)/sizeof(save_these[0]); i++) {
    double tm = find_time(save_these[i].key);
    printf("\\gdef\\%s{%.2f}\n", save_these[i].macroname, tm);
  }

  return 0;
}
