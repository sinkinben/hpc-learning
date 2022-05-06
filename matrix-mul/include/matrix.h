#ifndef __MATRIX_H__
#define __MATRIX_H__
#include <cassert>
#include <utility>
#include <iostream>
#include <cstdlib>
struct Matrix
{
    int rows, cols;
    float *elements;
    Matrix(int m, int n) : rows(m), cols(n), elements((float *)aligned_alloc(32, sizeof(float) * m * n)) {}

    void Destroy() { delete[] elements; }

    size_t Size() const { return (size_t)rows * cols; }

    void Randomize()
    {
        for (int i = 0; i < rows * cols; ++i)
            elements[i] = (rand() / RAND_MAX);
    }

    void Fill(float val = 0)
    {
        std::fill(elements, elements + rows * cols, val);
    }

    inline float &Get(int i, int j) const
    {
        /* assert(0 <= i && i < rows && 0 <= j && j < cols); */
        return elements[i * cols + j];
    }
};
#endif