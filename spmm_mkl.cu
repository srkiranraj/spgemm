#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include "/opt/intel/mkl/include/mkl.h"
#include "/opt/intel/mkl/include/mkl_spblas.h"


#include "custom/print.h"
#include "custom/timer.h"

int main()
{
	int i = 0;

	char ta = 'n';
	int r[] = {0, 1, 2};
	int sort = 3;
	int info;

	int m, n, k, nzmax;

	float *a, *c;
	int *ia, *ja, *ic, *jc;

	// GpuTimer timer;
	double mkltime;

	// read mtx matrix into a csr_matrix
	cusp::csr_matrix<int, float, cusp::host_memory> A;
	cusp::io::read_matrix_market_file(A, "./Inputs/amazon0312.mtx");
	//cusp::io::read_matrix_market_file(A, "./Inputs/sample.mtx");

	a = (float *)malloc((A.row_offsets[A.num_rows]) * sizeof(float));
	ja = (int *)malloc((A.row_offsets[A.num_rows]) * sizeof(int));
	ia = (int *)malloc((A.num_rows + 1) * sizeof(int));

	for (i = 0; i < A.num_rows + 1; ++i)
		ia[i] = A.row_offsets[i] + 1;

	for (i = 0; i < A.row_offsets[A.num_rows]; ++i)
	{
		a[i] = A.values[i];
		ja[i] = A.column_indices[i] + 1;
	}


	ic = (int *)malloc((A.num_rows+1) * sizeof(int));
	m = A.num_rows;

	printf("... Start ...\n");
	mkltime = omp_get_wtime();

	mkl_scsrmultcsr(&ta, &r[1], &sort, &m, &m, &m, a, ja, ia,  a, ja, ia, c, jc, ic, &nzmax, &info);

	//printarray(ic, 0, A.num_rows+1, "\n");
	
	c = (float *)malloc((ic[A.num_rows] - 1) * sizeof(float));
	jc = (int *)malloc((ic[A.num_rows] - 1) * sizeof(int));

	mkl_scsrmultcsr(&ta, &r[2], &sort, &m, &m, &m, a, ja, ia, a, ja, ia, c, jc, ic, &nzmax, &info);
	
	//printfloatarray(c, 0, ic[A.num_rows] - 1, "\n");

	//std::cout<<"DEBUG: "<< ic[A.num_rows]-1<<"\n";	
	std::cout<< (omp_get_wtime()-mkltime)<< " Second(s)\n";
	printf("... End ... \n");

	return 0;
}
