#include "matrix.h"
#include <algorithm>
#include <thread>
#include <vector>
static constexpr int TILE = 2;
static constexpr int THREADS = 4; // There are 4 cores on my machine.

static inline void ComputeTile(Matrix A, Matrix B, Matrix C, int x, int y)
{
    for (int i = x; i < x + TILE; ++i)
    {
        for (int j = y; j < y + TILE; ++j)
        {
            for (int idx = y; idx < y + TILE; ++idx)
                C.Get(i, j) += A.Get(i, idx) * B.Get(idx, j);
        }
    }
}

void MatrixMul(Matrix A, Matrix B, Matrix C)
{
    std::vector<std::thread> pool;
    auto task = [&](int x0, int y0, int x1, int y1)
    {
        for (int i = x0; i < x1; i += TILE)
            for (int j = y0; j < y1; j += TILE)
                ComputeTile(A, B, C, i, j);
    };
    int m = C.rows, k = C.cols;
    pool.emplace_back(std::thread(task, 0, m / 2, 0, k / 2));
    pool.emplace_back(std::thread(task, 0, m / 2, k / 2, k));
    pool.emplace_back(std::thread(task, m / 2, 0, 0, k / 2));
    pool.emplace_back(std::thread(task, m / 2, 0, k / 2, 0));

    for (auto &t : pool)
        t.join();
}
