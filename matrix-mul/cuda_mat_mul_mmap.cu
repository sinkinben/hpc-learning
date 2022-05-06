/* Matrix Multiplication on CUDA, based on block shared memory and memory mapping. */
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
void MatrixMul(Matrix A, Matrix B, Matrix C)
{
    int m = A.rows, n = A.cols, k = C.cols;
    /* allocate mapping memory */
    Matrix da(m, n, Memory::EMPTY);
    Matrix db(n, k, Memory::EMPTY);
    Matrix dc(m, k, Memory::EMPTY);

    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 gridDim(m / blockDim.x + (m % blockDim.x != 0),
                 k / blockDim.y + (k % blockDim.y != 0));

    /* create matrice accessed by device */
    assert(cudaHostGetDevicePointer(&da.elements, A.elements, 0) == cudaSuccess);
    assert(cudaHostGetDevicePointer(&db.elements, B.elements, 0) == cudaSuccess);
    assert(cudaHostGetDevicePointer(&dc.elements, C.elements, 0) == cudaSuccess);

    DeviceMatrixMul<<<gridDim, blockDim>>>(da, db, dc);

    /* let host wait for device to finish its job */
    cudaDeviceSynchronize();
}
