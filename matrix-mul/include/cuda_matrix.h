#ifndef __CUDA_MATRIX_H__
#define __CUDA_MATRIX_H__

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdio>

#define CUDA_BLOCK_SIZE (32)

enum Memory
{
    EMPTY = -1,
    Device = 0,
    Host = 1,
    MemMap = 2
};

struct CudaMatrix
{
    int8_t reside;
    int rows, cols;
    float *elements;

    /* Common Matrix Constructor */
    __host__ CudaMatrix(int m, int n, int8_t reside = Memory::Host) : rows(m), cols(n), reside(reside), elements(nullptr)
    {
        if (reside == Memory::Host)
            elements = new float[m * n];
        else if (reside == Memory::Device)
            assert(cudaMalloc(&elements, m * n * sizeof(float)) == cudaSuccess);
        else if (reside == Memory::MemMap)
            assert(cudaHostAlloc(&elements, m * n * sizeof(float), 0) == cudaSuccess);
    }
    void Destroy()
    {
        assert(elements != nullptr);
        if (reside == Memory::Host)
            delete[] elements;
        else if (reside == Memory::Device)
            assert(cudaFree(elements) == cudaSuccess);
        else if (reside == Memory::MemMap)
            assert(cudaFreeHost(elements) == cudaSuccess);
        elements = nullptr;
    }
    void Randomize()
    {
        for (int i = 0; i < rows * cols; ++i)
            elements[i] = (float)rand() / RAND_MAX;
    }
    void Fill(float val = 0)
    {
        std::fill(elements, elements + rows * cols, val);
    }

    /* Return element Matrix[i, j] */
    __host__ __device__ float &Get(int i, int j) const
    {
        assert(0 <= i && i < rows && 0 <= j && j < cols);
        return elements[i * cols + j];
    }

    size_t Size() const { return (size_t)rows * cols; }
};

using Matrix = CudaMatrix;

#endif