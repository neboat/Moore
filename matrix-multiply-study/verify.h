#include <stdbool.h>
#include <assert.h>
#include <string.h>

static bool need_to_verify=false;
static void parse_args(int argc, __const__  char *argv[]) {
  if (argc==2) {
    assert(strcmp(argv[1], "--verify")==0);
    need_to_verify=true;
  } else {
    assert(argc==1);
  }
}

static bool close_enough(double x, double y) {
    x = x<0 ? -x : x;
    y = y<0 ? -y : y;
    if (x<y) {
	double T = x;
	x = y;
	y = T;
    }
    double diff = x-y;
    assert(diff>=0);
    return (diff<0.01) || (diff/x < 0.00001);
}

static void verify(void) {
  for (int i=0; i<n; ++i) {         
    for (int j=0; j<n; ++j) {       
      double sum = 0;
      for (int k=0; k<n; ++k) {     
        sum += A[i][k] * B[k][j];
      }
      if (!close_enough(sum, C[i][j])) {
	printf("C[%d][%d] is %12.5g but should be %12.5g\n", i, j, C[i][j], sum);
	assert(false);
      }
    }
  }
}
