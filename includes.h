// C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>


// C++
#include <iostream>

// OPEN MP
#include <omp.h>

// MKL
#include "/opt/intel/mkl/include/mkl.h"
#include "/opt/intel/mkl/include/mkl_spblas.h"

// CUDA
#include <cuda_runtime.h>

// THRUST
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>

// CUSPARSE
#include <cusparse_v2.h>

// CUSP
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>

// CUSTOM
#include "custom/timer.h"
#include "custom/print.h"
