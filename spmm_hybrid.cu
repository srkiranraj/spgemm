#include "includes.h"
#include "conf.h"


int main(int argc, char *argv[])
{

	if(argc != 3)
	{
		printf("ERROR: Invalid number of arguments.\n");
		printf("USAGE: ./<output_file> <0 | 1 for debug> <input_filepath>\n");
		exit(1);
	}

    int debug = atoi(argv[1]);

	//GpuTimer timer;
	double phase1, phase2, phase3, phase4, total_time;

	int i = 0, j = 0;

	float interval;
	int num_bins;
	int *bins;
	int t;

	char ta = 'N';
	int r[] = {0, 1, 2};
	int sort = 3;
	int info;
	int m, n, k, nzmax;

	// switch on nested parallelism in OpenMP
	omp_set_nested(1);
	
	// phase 0
	// read mtx matrix into a csr_matrix
	cusp::csr_matrix<int, int, cusp::host_memory> A;
	cusp::io::read_matrix_market_file(A, argv[2]);

	// print input [csr] matrix
	if(debug)
	{
		std::cout<<"phase 0 -MTX to CSR Matrix Completed\n";
		cusp::print(A);
		cusp::print(A.row_offsets);
		cusp::print(A.values);
		cusp::print(A.column_indices);
	}
	
	// initialize 1d array to store nnz in each row
	thrust::host_vector<int> nnz(A.num_rows);	

	// compute nnz in each row
	#pragma omp parallel for
	for(i = 0; i < A.num_rows; i++)
		nnz[i] = A.row_offsets[i+1] - A.row_offsets[i];

	// dump nnz into a file
	if(debug)
	{
		FILE *fp;
		fp = fopen("./Outputs/nnz_dump.txt", "w");
		for(i = 0; i < A.num_rows; i++)
			fprintf(fp, "%d\t", nnz[i]);
		fclose(fp); 
	}

	// compute minmax of nnz
	thrust::pair<int *, int *> minmax = thrust::minmax_element(thrust::raw_pointer_cast(&(nnz[0])), thrust::raw_pointer_cast(&(nnz[0])) + A.num_rows);
	num_bins = sqrt(*minmax.second - *minmax.first);
	bins = (int *)calloc((num_bins+1), sizeof(int));
	interval = (float)(*minmax.second - *minmax.first)/num_bins;	

	// print overall bin information
	if(debug)
	{
		std::cout<<"phase I - Binning Information\n";
		std::cout<<"Max NNZ = "<<*minmax.second<<"\n";	
		std::cout<<"Min NNZ = "<<*minmax.first<<"\n";
		std::cout<<"Number of Bins = "<<num_bins+1<<"\n";
		std::cout<<"Width of Bins = "<<interval<<"\n";
	}

	// phase I
	// cpu
	// binning nnz to find optimal "t
	phase1 = omp_get_wtime();
	#pragma omp parallel for
	for(i = 0; i < A.num_rows; i++)
	{
		int index = (int)((nnz[i] - *minmax.first)/interval);
		#pragma omp atomic
			bins[index]++;
	}
	
	if(debug)
		assert(printarraywithsum(bins, 0, (num_bins + 1), "\n") == A.num_rows);


	// calculate optimal "t" from bins
	// todo
	// temporarily hardcoded
	t = 2;
	
	// cpu
	// obtaining matrices Al and Ah
	int l_rp = 0, h_rp = 0;

	// allocate space for Al, Ah, AhxAh, Al*Al sub-matrix
	// rows = m
	// values & columns = nnz of A [in worst case]
	float *Al = (float *)malloc((A.row_offsets[A.num_rows]) * sizeof(float));
	int *j_Al = (int *)malloc((A.row_offsets[A.num_rows]) * sizeof(int));
	int *i_Al = (int *)malloc((A.num_rows + 1) * sizeof(int));

	float *Ah = (float *)malloc((A.row_offsets[A.num_rows]) * sizeof(float));
	int *j_Ah = (int *)malloc((A.row_offsets[A.num_rows]) * sizeof(int));
	int *i_Ah = (int *)malloc((A.num_rows + 1) * sizeof(int));

	float *AhxAh;
	int *j_AhxAh;
	int *i_AhxAh = (int *)malloc((A.num_rows + 1) * sizeof(int));

	float *AlxAl;
	int *j_AlxAl;
	int *i_AlxAl = (int *)malloc((A.num_rows + 1) * sizeof(int));

	float *AhxAl;
	int *j_AhxAl;
	int *i_AhxAl = (int *)malloc((A.num_rows + 1) * sizeof(int));

	float *AlxAh;
	int *j_AlxAh;
	int *i_AlxAh = (int *)malloc((A.num_rows + 1) * sizeof(int));

	float *d_Al, *d_AlxAl;
	int *d_j_Al, *d_j_AlxAl;
	int *d_i_Al, *d_i_AlxAl;

	float *d_Ah, *d_AhxAl;
	int *d_j_Ah, *d_j_AhxAl;
	int *d_i_Ah, *d_i_AhxAl;

	
	for(i = 0; i < A.num_rows; i++)
	{
		i_Al[i] = l_rp+1;
		i_Ah[i] = h_rp+1;

		// copy row into Al if nnz is less than "t"
		if(nnz[i] <= t)
		{
			#pragma omp parallel for
			for(j = 0; j < nnz[i]; j++)
			{
				j_Al[l_rp + j] = A.column_indices[A.row_offsets[i] + j] + 1;
				Al[l_rp + j] = A.values[A.row_offsets[i] + j];
			}
			l_rp = l_rp + nnz[i];
		}
		else
		{
			#pragma omp parallel for
			for (j = 0; j < nnz[i]; j++)
			{
				j_Ah[h_rp + j] = A.column_indices[A.row_offsets[i] + j] + 1;
				Ah[h_rp + j] = A.values[A.row_offsets[i] + j];
			}
			h_rp = h_rp + nnz[i];
		}
	}
	i_Al[i] = l_rp+1;
	i_Ah[i] = h_rp+1;


	// allocation and copy for AlxAl
	cudaMalloc((void**)&d_Al, sizeof(float) * (i_Al[A.num_rows] - 1));
 	cudaMalloc((void**)&d_j_Al, sizeof(int) * (i_Al[A.num_rows] - 1));
 	cudaMalloc((void**)&d_i_Al, sizeof(int) * (A.num_rows + 1));
    
    cudaMalloc((void**)&d_i_AlxAl, sizeof(int) * (A.num_rows + 1));

    cudaMemcpy(d_Al, Al, sizeof(float) * (i_Al[A.num_rows] - 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_j_Al, j_Al, sizeof(int) * (i_Al[A.num_rows] - 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_Al, i_Al, sizeof(int) * (A.num_rows + 1), cudaMemcpyHostToDevice); 


	// allocation and copy for AhxAl
    cudaMalloc((void**)&d_Ah, sizeof(float) * (i_Ah[A.num_rows] - 1));
 	cudaMalloc((void**)&d_j_Ah, sizeof(int) * (i_Ah[A.num_rows] - 1));
 	cudaMalloc((void**)&d_i_Ah, sizeof(int) * (A.num_rows + 1));

    cudaMalloc((void**)&d_i_AhxAl, sizeof(int) * (A.num_rows + 1));

    cudaMemcpy(d_Ah, Ah, sizeof(float) * (i_Ah[A.num_rows] - 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_j_Ah, j_Ah, sizeof(int) * (i_Ah[A.num_rows] - 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_Ah, i_Ah, sizeof(int) * (A.num_rows + 1), cudaMemcpyHostToDevice); 
	
    // cudaThreadSynchronize();

	std::cout<<"PHASE I :"<<phase1 - omp_get_wtime()<<" Second(s)\n";

	// todo
	// Reallocate space for Al, j_Al using l_rp
	// Reallocate space for Ah, j_Ah using h_rp

	// print sub-matrices Al, Ah
	if(debug)
	{
		std::cout<<"Sub-Matrix Al\n";
		printarray(i_Al, 0, A.num_rows+1, "\n");
		printtwoarray(j_Al, Al, 0, i_Al[A.num_rows] - 1, "\t", "\n");

		std::cout<<"Sub-Matrix Ah\n";
		printarray(i_Ah, 0, A.num_rows+1, "\n");
		printtwoarray(j_Ah, Ah, 0, i_Ah[A.num_rows] - 1, "\t", "\n");
	}

	m = A.num_rows;
	n = A.num_cols;
	k = A.num_cols;
	nzmax = m * k;

	// phase II
	phase2 = omp_get_wtime();

	#pragma omp parallel num_threads(2)
	{
		int ID = omp_get_thread_num();
		if(ID == 0)
		{
			// cpu
			// Ah * Ah		
			mkl_scsrmultcsr(&ta, &r[1], &sort, &m, &n, &k, Ah, j_Ah, i_Ah, Ah, j_Ah, i_Ah, AhxAh, j_AhxAh, i_AhxAh, &nzmax, &info);

		    AhxAh = (float *)malloc((i_AhxAh[m]-1) * sizeof(float));
			j_AhxAh = (int *)malloc((i_AhxAh[m]-1) * sizeof(int));

			mkl_scsrmultcsr(&ta, &r[2], &sort, &m, &n, &k, Ah, j_Ah, i_Ah, Ah, j_Ah, i_Ah, AhxAh, j_AhxAh, i_AhxAh, &nzmax, &info);
			

			
		}
		else if(ID == 1)
		{
			// gpu
			// Al * Al

			// cusparse
		    cusparseHandle_t handle = 0;
		    cusparseMatDescr_t descrAl = 0;
		    cusparseMatDescr_t descrAlxAl = 0;

		    // initialize cusparse library
		    cusparseCreate(&handle);
		    
		    // create and setup matrix descriptors A, B & C
		    cusparseCreateMatDescr(&descrAl);        
		    cusparseSetMatType(descrAl, CUSPARSE_MATRIX_TYPE_GENERAL);
		    cusparseSetMatIndexBase(descrAl, CUSPARSE_INDEX_BASE_ONE);  

		    cusparseCreateMatDescr(&descrAlxAl);
		    cusparseSetMatType(descrAlxAl, CUSPARSE_MATRIX_TYPE_GENERAL);
		    cusparseSetMatIndexBase(descrAlxAl, CUSPARSE_INDEX_BASE_ONE);  

		    int baseC, nnzC;
		    int *nnzTotalDevHostPtr = &nnzC;
		    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

		    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, A.num_rows, A.num_rows, A.num_rows, 
                        		descrAl, i_Al[A.num_rows] - 1, d_i_Al, d_j_Al,
                        		descrAl, i_Al[A.num_rows] - 1, d_i_Al, d_j_Al,
                        		descrAlxAl, d_i_AlxAl, nnzTotalDevHostPtr);
		    
		    cudaMemcpy(&nnzC , d_i_AlxAl + A.num_rows, sizeof(int), cudaMemcpyDeviceToHost);
		    cudaMemcpy(&baseC, d_i_AlxAl, sizeof(int), cudaMemcpyDeviceToHost);

		    nnzC = nnzC - baseC;
		    // allocate memory according to nnzC
		    
		    cudaMalloc((void**)&d_AlxAl, sizeof(float) * nnzC);
		    cudaMalloc((void**)&d_j_AlxAl, sizeof(int) * nnzC);

		    cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, A.num_rows, A.num_rows, A.num_rows,
		                     descrAl, i_Al[A.num_rows] - 1,
		                     d_Al, d_i_Al, d_j_Al,
		                     descrAl, i_Al[A.num_rows] - 1,
		                     d_Al, d_i_Al, d_j_Al,
		                     descrAlxAl,
		                     d_AlxAl, d_i_AlxAl, d_j_AlxAl);

		    AlxAl = (float *)malloc(nnzC * sizeof(float));
    		j_AlxAl = (int *)malloc(nnzC * sizeof(int));

    		cudaMemcpy(AlxAl, d_AlxAl, sizeof(float) * nnzC, cudaMemcpyDeviceToHost);
			cudaMemcpy(j_AlxAl, d_j_AlxAl, sizeof(int) * nnzC, cudaMemcpyDeviceToHost);
			cudaMemcpy(i_AlxAl, d_i_AlxAl, sizeof(int) * (A.num_rows + 1), cudaMemcpyDeviceToHost);

		}
	}

	std::cout<< (omp_get_wtime()-phase2) << " Second(s)\n";

	if(debug)
	{
		std::cout<<"phase II :: partial result -> AhxAh\n";
		printarray(i_AhxAh, 0, A.num_rows+1, "\n");
		printtwoarray(j_AhxAh, AhxAh, 0, (i_AhxAh[m] - 1), "\t", "\n");

		printarray(i_AlxAl, 0, A.num_rows+1, "\n");
		printtwoarray(j_AlxAl, AlxAl, 0, (i_AlxAl[m] - 1), "\t", "\n");
	}

	// phase III
	phase3 = omp_get_wtime();

	#pragma omp parallel num_threads(2)
	{
		int ID = omp_get_thread_num();
		if(ID == 0)
		{
			// cpu
			// Ah * Ah		
			mkl_scsrmultcsr(&ta, &r[1], &sort, &m, &n, &k, Al, j_Al, i_Al, Ah, j_Ah, i_Ah, AlxAh, j_AlxAh, i_AlxAh, &nzmax, &info);

		    AlxAh = (float *)malloc((i_AlxAh[m]-1) * sizeof(float));
			j_AlxAh = (int *)malloc((i_AlxAh[m]-1) * sizeof(int));

			mkl_scsrmultcsr(&ta, &r[2], &sort, &m, &n, &k, Al, j_Al, i_Al, Ah, j_Ah, i_Ah, AlxAh, j_AlxAh, i_AlxAh, &nzmax, &info);
			
		}
		else if(ID == 1)
		{
			// gpu
			// Al * Al

			// cusparse
		    cusparseHandle_t handle = 0;
		    cusparseMatDescr_t descrAh = 0;
		    cusparseMatDescr_t descrAl = 0;
		    cusparseMatDescr_t descrAhxAl = 0;

		    // initialize cusparse library
		    cusparseCreate(&handle);
		    
		    // create and setup matrix descriptors A, B & C
		    cusparseCreateMatDescr(&descrAh);        
		    cusparseSetMatType(descrAh, CUSPARSE_MATRIX_TYPE_GENERAL);
		    cusparseSetMatIndexBase(descrAh, CUSPARSE_INDEX_BASE_ONE);  

		    cusparseCreateMatDescr(&descrAl);        
		    cusparseSetMatType(descrAl, CUSPARSE_MATRIX_TYPE_GENERAL);
		    cusparseSetMatIndexBase(descrAl, CUSPARSE_INDEX_BASE_ONE);  

		    cusparseCreateMatDescr(&descrAhxAl);
		    cusparseSetMatType(descrAhxAl, CUSPARSE_MATRIX_TYPE_GENERAL);
		    cusparseSetMatIndexBase(descrAhxAl, CUSPARSE_INDEX_BASE_ONE);  

		    int baseC, nnzC;
		    int *nnzTotalDevHostPtr = &nnzC;
		    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

		    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, A.num_rows, A.num_rows, A.num_rows, 
                        		descrAh, i_Ah[A.num_rows] - 1, d_i_Ah, d_j_Ah,
                        		descrAl, i_Al[A.num_rows] - 1, d_i_Al, d_j_Al,
                        		descrAhxAl, d_i_AhxAl, nnzTotalDevHostPtr);
		    
		    cudaMemcpy(&nnzC , d_i_AhxAl + A.num_rows, sizeof(int), cudaMemcpyDeviceToHost);
		    cudaMemcpy(&baseC, d_i_AhxAl, sizeof(int), cudaMemcpyDeviceToHost);

		    nnzC = nnzC - baseC;
		    // allocate memory according to nnzC
		    
		    cudaMalloc((void**)&d_AhxAl, sizeof(float) * nnzC);
		    cudaMalloc((void**)&d_j_AhxAl, sizeof(int) * nnzC);

		    cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, A.num_rows, A.num_rows, A.num_rows,
		                     descrAh, i_Ah[A.num_rows] - 1,
		                     d_Ah, d_i_Ah, d_j_Ah,
		                     descrAl, i_Al[A.num_rows] - 1,
		                     d_Al, d_i_Al, d_j_Al,
		                     descrAhxAl,
		                     d_AhxAl, d_i_AhxAl, d_j_AhxAl);

		    AhxAl = (float *)malloc(nnzC * sizeof(float));
    		j_AhxAl = (int *)malloc(nnzC * sizeof(int));

    		cudaMemcpy(AhxAl, d_AhxAl, sizeof(float) * nnzC, cudaMemcpyDeviceToHost);
			cudaMemcpy(j_AhxAl, d_j_AhxAl, sizeof(int) * nnzC, cudaMemcpyDeviceToHost);
			cudaMemcpy(i_AhxAl, d_i_AhxAl, sizeof(int) * (A.num_rows + 1), cudaMemcpyDeviceToHost);

		}
	}

	std::cout<< (omp_get_wtime()-phase3) << " Second(s)\n";	
	
	if(debug)
	{
		std::cout<<"phase II :: partial result -> AhxAh\n";
		printarray(i_AlxAh, 0, A.num_rows+1, "\n");
		printtwoarray(j_AlxAh, AlxAh, 0, (i_AlxAh[m] - 1), "\t", "\n");

		printarray(i_AhxAl, 0, A.num_rows+1, "\n");
		printtwoarray(j_AhxAl, AhxAl, 0, (i_AhxAl[m] - 1), "\t", "\n");
	}

	return 0;
}
