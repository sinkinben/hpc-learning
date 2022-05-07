/* GEMM based on ISPC-SPMD */
#include <iostream>
#include <cassert>
#include <cmath>
#include "config.h"
#include "spmd.h"
#define Get(array, i, j, cols) (array[i * cols + j])

static inline void Fill(float arr[], int size)
{
    for (int i = 0; i < size; ++i)
        arr[i] = rand() / RAND_MAX;
}

inline void Check(int m, int n, int k, float *A, float *B, float *C)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            float expect = 0;
            for (int idx = 0; idx < n; ++idx)
                expect += Get(A, i, idx, n) * Get(B, idx, j, k);
            assert(fabs(Get(C, i, j, k) - expect) < 1e-4);
        }
    }
}

static inline void RunTest(int m, int n, int k, const char *name)
{
    float *A = new float[m * n];
    float *B = new float[n * k];
    float *C = new float[m * k];
    Fill(A, m * n);
    Fill(B, n * k);

    Timer timer;
    uint64_t cycleStart = 0;
    double totalTime = 0;
    uint64_t totalCycles = 0;
    for (int i = 0; i < TEST; ++i)
    {
        timer.reset();
        cycleStart = Benchmark::GetCPUCycle();
        ispc::MatrixMul(m, n, k, A, B, C);
        totalCycles += Benchmark::GetCPUCycle() - cycleStart;
        totalTime += timer.elapsed_nano();
    }
    PrintResult(name, ((totalTime / 1e6) / TEST), (double)totalCycles / TEST);
#ifdef OPEN_CHECKING
    Check(m, n, k, A, B, C);
#endif
    delete[] A;
    delete[] B;
    delete[] C;
}

int main(int argc, char *argv[])
{
    RunTest(SIZE, SIZE, SIZE, argv[0] + 2);
}