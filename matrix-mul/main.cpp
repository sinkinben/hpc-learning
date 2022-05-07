
#include <iostream>
#include <cmath>
#include "config.h"

#ifdef CUDA
#include "cuda_matrix.h"
#else
#include "matrix.h"
#endif

extern void MatrixMul(Matrix, Matrix, Matrix);

static inline void Check(Matrix A, Matrix B, Matrix C)
{
    for (int i = 0; i < C.rows; ++i)
    {
        for (int j = 0; j < C.cols; ++j)
        {
            float expect = 0;
            for (int idx = 0; idx < A.cols; ++idx)
                expect += A.Get(i, idx) * B.Get(idx, j);
            assert(fabs(C.Get(i, j) - expect) < 1e-4);
        }
    }
}

inline void RunTest(int m, int n, int k, const char *name)
{
#ifdef CUDA_MMAP
    Matrix A(m, n, Memory::MemMap);
    Matrix B(n, k, Memory::MemMap);
    Matrix C(m, k, Memory::MemMap);
#else
    Matrix A(m, n);
    Matrix B(n, k);
    Matrix C(m, k);
#endif
    A.Randomize(), B.Randomize();

    Timer timer;
    uint64_t cycleStart = 0;
    double totalTime = 0;
    uint64_t totalCycles = 0;
    for (int i = 0; i < TEST; ++i)
    {
        C.Fill(0);
        timer.reset();
        cycleStart = Benchmark::GetCPUCycle();
        MatrixMul(A, B, C);
        totalCycles += Benchmark::GetCPUCycle() - cycleStart;
        totalTime += timer.elapsed_nano();
    }
    PrintResult(name, ((totalTime / 1e6) / TEST), (double)totalCycles / TEST);
#ifdef OPEN_CHECKING
    Check(A, B, C);
#endif
    A.Destroy(), B.Destroy(), C.Destroy();
}

int main(int argc, char *argv[])
{
    std::ios::sync_with_stdio(0);
    RunTest(SIZE, SIZE, SIZE, argv[0] + 2);
}