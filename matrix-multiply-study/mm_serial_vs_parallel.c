// Copyright 2013 Bradley C. Kuszmaul, Charles E. Leiserson, and Tao B. Schardl
/* Create mm_serial_vs_parallel.tex */
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
#include "timedata.c"
  {0, 0, 0, 0, 0}
};

static double find_time(const char *implementation) {
  int count = 0;
  double best = 1e30;
  for (int i = 0; tinfos[i].machine; i++) {
    if ((strcmp(tinfos[i].machine, "alf") == 0)
        && (strcmp(tinfos[i].implementation, implementation) == 0)) {
      count++;
      double thist = tinfos[i].runtime;
      if (thist < best) best=thist;
    }
  }
  if (count == 0) return -1;
  else            return best;
}

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
  struct rowinfo names[]={{true,  "mm_vec",                         "Vectorized serial loops"},
                          {true,  "mm_tile",                        "+ tiling"},
                          {true,  "mm_tile_transpose",              "+ transpose"},
                          {true,  "mm_sdac",                        "Divide-and-conquer"},
                          {true,  "mm_sdac_coarsen",                "+ coarsening"},
                          {true,  "mm_sdac_coarsen_transpose",      "+ transpose"},
                          {true,  "mm_ploops",                      "Parallel loops"},
                          {true,  "mm_ploops_tile",                 "+ tiling"},
                          {true, "mm_ptile_transpose",             "+ transpose"},
                          {true,  "mm_dac",                         "Parallel divide-and-conquer"},
                          {true,  "mm_dac_coarsen",                 "+ coarsening"},
                          {true,  "mm_dac_coarsen_transpose",       "+ transpose"},
                          {false, 0, 0}};
  for (int i = 0; names[i].key; i++) {
    do_row(&names[i]);
  }
}

int main(int argc __attribute__((__unused__)), char *argv[]  __attribute__((__unused__))) {
  printf("%% -*- mode: latex; tex-fontify-script: nil; -*-\n");
  printf("\\begin{tabular}[t]{|d{2.0}ld{5.2}d{3.3}d{5.1}d{2.1}d{2.3}|}%%\n");
  printf("\\hline\n");
  printf("\\multicolumn{1}{|c}{\\textit{Version}}\n");
  printf(" & \\multicolumn{1}{l}{\\textit{Implementation}}\n");
  {
    char *names[][2]={{"Time (s)", ""},
                      {"GFLOPS",   ""},
                      {"Absolute", "speedup"},
                      {"Relative", "speedup"},
                      {"Percent",  "of peak"},
                      {0, 0}};
    int i;
    for (i = 0; names[i+1][0]; i++) {
      printf(" &\\multicolumn{1}{c}{\\textit{%s}}\n", names[i][0]);
    }
    printf(" &\\multicolumn{1}{c|}{\\textit{%s}}\n", names[i][0]);
    printf("\\\\[-2pt] &");
    for (int i = 0; names[i+1][0]; i++) {
      printf(" &\\multicolumn{1}{c}{\\textit{%s}}\n", names[i][1]);
    }
    printf(" &\\multicolumn{1}{c|}{\\textit{%s}}\n", names[i][1]);
    printf("\\\\\n");
    printf("\\hline\n");
  }
  do_rows();
    printf("\\hline\n");
  printf("\\end{tabular}%%\n");

  return 0;
}
