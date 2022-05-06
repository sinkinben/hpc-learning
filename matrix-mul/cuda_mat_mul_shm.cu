#include "cuda_matrix.h"
__global__ void DeviceMatrixMul(Matrix A, Matrix B, Matrix C)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.x * CUDA_BLOCK_SIZE + tx;
    int col = blockIdx.y * CUDA_BLOCK_SIZE + ty;

    /* result of C[i, j] */
    float res = 0;

    /* shared memory within block */
    __shared__ float sharedA[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
    __shared__ float sharedB[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

    for (int k = 0; k < A.cols; k += CUDA_BLOCK_SIZE)
    {

        sharedA[tx][ty] = (row < A.rows && k + ty < A.cols) ? A.Get(row, k + ty) : 0;
        sharedB[tx][ty] = (k + tx < B.rows && col < B.cols) ? B.Get(k + tx, col) : 0;
        __syncthreads();

        for (int j = 0; j < CUDA_BLOCK_SIZE; ++j)
            res += sharedA[tx][j] * sharedB[j][ty];
        __syncthreads();
    }
    if (row < C.rows && col < C.cols)
        C.Get(row, col) = res;
}

__host__ void MatrixMul(Matrix A, Matrix B, Matrix C)
{
    /* check validity of input matrice */
    assert(A.cols == B.rows && C.rows == A.rows && C.cols == B.cols);

    /* matrice in device memory, prefix 'd' means 'device' */
    Matrix da(A.rows, A.cols, Memory::Device);
    Matrix db(B.rows, B.cols, Memory::Device);
    Matrix dc(C.rows, C.cols, Memory::Device);

    cudaMemcpy(da.elements, A.elements, sizeof(float) * A.Size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db.elements, B.elements, sizeof(float) * B.Size(), cudaMemcpyHostToDevice);

    /* One thread compute one element C[i, j].
     * Max number of threads per block is 1024 = 32 x 32.
     */
    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 gridDim(C.rows / CUDA_BLOCK_SIZE + (C.rows % CUDA_BLOCK_SIZE != 0),
                 C.cols / CUDA_BLOCK_SIZE + (C.cols % CUDA_BLOCK_SIZE != 0));

    DeviceMatrixMul<<<gridDim, blockDim>>>(da, db, dc);

    /* let host wait for device to finish its job */
    cudaDeviceSynchronize();

    /* copy result from CUDA to host */
    cudaMemcpy(C.elements, dc.elements, sizeof(float) * C.Size(), cudaMemcpyDeviceToHost);
    da.Destroy(), db.Destroy(), dc.Destroy();
}
