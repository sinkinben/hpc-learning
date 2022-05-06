#include "cuda_matrix.h"

__global__ void DeviceMatrixMul(Matrix a, Matrix b, Matrix c)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < c.rows && col < c.cols)
    {
        float res = 0;
        for (int j = 0; j < a.cols; ++j)
            res += a.Get(row, j) * b.Get(j, col);
        c.Get(row, col) = res;
    }
}

__host__ void MatrixMul(Matrix A, Matrix B, Matrix C)
{
    /* check validity of input matrice */
    assert(A.cols == B.rows && C.rows == A.rows && C.cols == B.cols);

    /* matrice in device memory */
    Matrix da(A.rows, A.cols, Memory::Device);
    Matrix db(B.rows, B.cols, Memory::Device);
    Matrix dc(C.rows, C.cols, Memory::Device);

    cudaMemcpy(da.elements, A.elements, sizeof(float) * A.Size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db.elements, B.elements, sizeof(float) * B.Size(), cudaMemcpyHostToDevice);

    /* One thread compute one element C[i, j].
     * Max number of threads per block is 1024.
     */
    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 gridDim(C.rows / blockDim.x + (C.rows % blockDim.x != 0),
                 C.cols / blockDim.y + (C.cols % blockDim.y != 0));
    DeviceMatrixMul<<<gridDim, blockDim>>>(da, db, dc);

    /* wait for device to finish its job */
    cudaDeviceSynchronize();
    cudaMemcpy(C.elements, dc.elements, sizeof(float) * C.Size(), cudaMemcpyDeviceToHost);
    da.Destroy(), db.Destroy(), dc.Destroy();
}