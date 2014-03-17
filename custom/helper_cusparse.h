#define CLEANUP(s)
do {
    printf ("%s\n", s);
    if (yHostPtr)           free(yHostPtr);
    if (zHostPtr)           free(zHostPtr);
    if (xIndHostPtr)        free(xIndHostPtr);
    if (xValHostPtr)        free(xValHostPtr);
    if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);
    if (cooColIndexHostPtr) free(cooColIndexHostPtr);
    if (cooValHostPtr)      free(cooValHostPtr);
    if (y)                  cudaFree(y);
    if (z)                  cudaFree(z);
    if (xInd)               cudaFree(xInd);
    if (xVal)               cudaFree(xVal);
    if (csrRowPtr)          cudaFree(csrRowPtr);
    if (cooRowIndex)        cudaFree(cooRowIndex);
    if (cooColIndex)        cudaFree(cooColIndex);
    if (cooVal)             cudaFree(cooVal);
    if (descr)              cusparseDestroyMatDescr(descr);
    if (handle)             cusparseDestroy(handle);
    cudaDeviceReset();
    fflush (stdout);
} while (0)
