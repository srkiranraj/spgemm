#include "includes.h"
#include "conf.h"


int main(int argc, char *argv[])
{
    int debug = atoi(argv[1]);

    int i = 0;

    // GPU Timer
    GpuTimer host_to_gpu_A;
    GpuTimer host_to_gpu_B;
    GpuTimer pre_spgemm;
    GpuTimer spgemm;
    GpuTimer gpu_to_host_C;

    // initialize matrix A, B & C
    cusp::csr_matrix<int, float, cusp::host_memory> A;
    cusp::csr_matrix<int, float, cusp::host_memory> B;
    cusp::csr_matrix<int, float, cusp::host_memory> C;

    // host pointers for A, B & C
    float *h_A, *h_B, *h_C;
    int *h_rA, *h_rB, *h_rC;
    int *h_cA, *h_cB, *h_cC;

    // device pointers for A, B & C
    float *d_A, *d_B, *d_C;
    int *d_rA, *d_rB, *d_rC;
    int *d_cA, *d_cB, *d_cC;

    // matrix dimensions & properties
    int m_original, m, n, k;
    int nnzA, nnzB, nnzC;

    // read matrix A & B
    cusp::io::read_matrix_market_file(A, argv[2]);
    B = A;

    // check if matrices are compatible for multiplication
    if(A.num_cols != B.num_rows)
    {
        perror("ERROR: Matrices not compatible for multiplication.");
        exit(1);
    }

    // initialize m_original, n, k
    m_original = A.num_rows;
    
    // check if m_original is divisible by ROW_BLOCK_SIZE
    if(m_original % ROW_BLOCK_SIZE != 0)
        m = ((m_original / ROW_BLOCK_SIZE) + 1) * ROW_BLOCK_SIZE;
    else
        m = m_original;

    n = B.num_rows;
    k = B.num_cols;

    // initialize nnzA, nnzB
    nnzA = A.row_offsets[m_original];
    nnzB = B.row_offsets[n];

    // allocate memory in host for matrices A, B, C
    h_A = (float *)malloc(nnzA * sizeof(float));
    h_cA = (int *)malloc(nnzA * sizeof(int));
    h_rA = (int *)malloc((m+1) * sizeof(int));

    h_B = (float *)malloc(nnzA * sizeof(float));
    h_cB = (int *)malloc(nnzA * sizeof(int));
    h_rB = (int *)malloc((m+1) * sizeof(int));

    h_rC = (int *)malloc((m+1) * sizeof(int));


    // A - cusp::csr_matrix -> 3-array format
    #pragma omp parallel for
    for (i = 0; i < (m_original+1); ++i)
        h_rA[i] = A.row_offsets[i] + 1;
    
    // extra loop to compensate for padding
    #pragma omp parallel for
    for (i = m_original + 1; i < (m+1); ++i)
        h_rA[i] = A.row_offsets[m_original] + 1;

    #pragma omp parallel for
    for (i = 0; i < nnzA; ++i)
    {
        h_cA[i] = A.column_indices[i] + 1;
        h_A[i] = A.values[i];
    }

    // B - cusp::csr_matrix -> 3-array format
    #pragma omp parallel for
    for (i = 0; i < (n+1); ++i)
        h_rB[i] = B.row_offsets[i] + 1;
    
    #pragma omp parallel for
    for (i = 0; i < nnzB; ++i)
    {
        h_cB[i] = B.column_indices[i] + 1;
        h_B[i] = B.values[i];
    }

    if(debug)
    {
        printf("... Matrix A ...\n");
        printfloatarray(h_A, 0, nnzA, "\n");
        printarray(h_cA, 0, nnzA, "\n");
        printarray(h_rA, 0, m+1, "\n");

        printf("... Matrix B ...\n");
        printfloatarray(h_B, 0, nnzB, "\n");
        printarray(h_cB, 0, nnzB, "\n");
        printarray(h_rB, 0, n+1, "\n");
    }

    // NOTE: row_pointers, column_indexes of A, B are now based on ONE-BASED INDEXING.

    // allocate menory for matrices A, B & C in device
    cudaMalloc((void**)&d_A, sizeof(float) * nnzA);
    cudaMalloc((void**)&d_cA, sizeof(int) * nnzA);
    cudaMalloc((void**)&d_rA, sizeof(int) * (m+1));

    cudaMalloc((void**)&d_B, sizeof(float) * nnzB);
    cudaMalloc((void**)&d_cB, sizeof(int) * nnzB);
    cudaMalloc((void**)&d_rB, sizeof(int) * (n+1));

    cudaMalloc((void**)&d_rC, sizeof(int) * (m+1));

    // cusparse
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descrA = 0;
    cusparseMatDescr_t descrB = 0;
    cusparseMatDescr_t descrC = 0;

    // initialize cusparse library
    cusparseCreate(&handle);
    
    // create and setup matrix descriptors A, B & C
    cusparseCreateMatDescr(&descrA);        
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);  

    cusparseCreateMatDescr(&descrB);        
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ONE);  

    cusparseCreateMatDescr(&descrC);        
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE);  

    // copy data from host to device
    host_to_gpu_A.Start();
    cudaMemcpy(d_A , h_A, sizeof(float) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cA , h_cA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rA , h_rA, sizeof(int) * (m + 1), cudaMemcpyHostToDevice); 
    host_to_gpu_A.Stop();
    std::cout<<"Time taken to copy data from CPU to GPU [For A]: "<<host_to_gpu_A.Elapsed()<<" Millisecond(s)\n";

    host_to_gpu_B.Start();
    cudaMemcpy(d_B , h_B, sizeof(float) * nnzB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cB , h_cB, sizeof(int) * nnzB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rB , h_rB, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);  
    host_to_gpu_B.Stop();
    std::cout<<"Time taken to copy data from CPU to GPU [For B]: "<<host_to_gpu_B.Elapsed()<<" Millisecond(s)\n";
    
    int baseC;
    int *nnzTotalDevHostPtr = &nnzC;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    // preliminary operation to know nnz in output matrix
    pre_spgemm.Start();
    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, 
                        descrA, nnzA, d_rA, d_cA,
                        descrB, nnzB, d_rB, d_cB,
                        descrC, d_rC, nnzTotalDevHostPtr );
    pre_spgemm.Stop();
    std::cout<<"Time taken for pre-spgemm step [nnzC compute]: "<<pre_spgemm.Elapsed()<<" Millisecond(s)\n";
    
    if (NULL != nnzTotalDevHostPtr)
    {
        nnzC = *nnzTotalDevHostPtr;
    }
    else
    {
        cudaMemcpy(&nnzC , d_rC + m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, d_rC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }

    // allocate memory according to nnzC
    cudaMalloc((void**)&d_C, sizeof(float) * nnzC);
    cudaMalloc((void**)&d_cC, sizeof(int) * nnzC);

    // actual spgemm is done here
    spgemm.Start();
    cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k,
                     descrA, nnzA,
                     d_A, d_rA, d_cA,
                     descrB, nnzB,
                     d_B, d_rB, d_cB,
                     descrC,
                     d_C, d_rC, d_cC);
    spgemm.Stop();
    std::cout<<"Time taken for spgemm step [C = A * B]: "<<spgemm.Elapsed()<<" Millisecond(s)\n";


    h_C = (float *)malloc(nnzC * sizeof(float));
    h_cC = (int *)malloc(nnzC * sizeof(int));

    // copy results back to host
    gpu_to_host_C.Start();
    cudaMemcpy(h_C , d_C, sizeof(float) * nnzC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cC , d_cC, sizeof(int) * nnzC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rC , d_rC, sizeof(int) * (m+1), cudaMemcpyDeviceToHost);
    gpu_to_host_C.Stop();
    std::cout<<"Time taken to send result back to CPU [For C]: "<<gpu_to_host_C.Elapsed()<<" Millisecond(s)\n";

    std::cout<<"Total Time: "<< host_to_gpu_A.Elapsed() + host_to_gpu_B.Elapsed() + pre_spgemm.Elapsed() + spgemm.Elapsed() + gpu_to_host_C.Elapsed() <<" Millisecond(s)\n\n";

    if(debug)
    {
        printf("... Matrix C ...\n");
        printfloatarray(h_C, 0, nnzC, "\n");
        printarray(h_cC, 0, nnzC, "\n");
        printarray(h_rC, 0, m+1, "\n");
    }
}
