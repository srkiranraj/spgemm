#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

#include <iostream>

#include "custom/timer.h"

int main(void)
{
    // initialize matrix
    cusp::csr_matrix<int, int, cusp::host_memory> A;
    cusp::io::read_matrix_market_file(A, "./Inputs/p2p-Gnutella04.mtx");

    GpuTimer timer;

    // allocate output matrix
    cusp::csr_matrix<int, int, cusp::host_memory> C(A.num_rows, A.num_cols, A.num_rows * A.num_cols);

    timer.Start();
    // compute y = A * x
    cusp::multiply(A, A, C);
    timer.Stop();

    std::cout<<timer.Elapsed()<<" Milli Second(s).\n";

    // print y
    // cusp::print(C);

    return 0;
}
