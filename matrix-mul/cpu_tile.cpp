#include "matrix.h"
#include <algorithm>

static constexpr int TILE = 2;

/* (x, y) is the index of one tile.
 * Please note that I don't check the bounder. 
 * The x + TILE, y + TILE may be out of index bounder.
 */
static inline void ComputeTile(Matrix A, Matrix B, Matrix C, int x, int y)
{
    for (int i = x; i < x + TILE; ++i)
    {
        for (int j = y; j < y + TILE; ++j)
        {
            float res = 0;
            for (int idx = y; idx < y + TILE; ++idx)
                res += A.Get(i, idx) * B.Get(idx, j);
            C.Get(i, j) += res;
        }
    }
}

void MatrixMul(Matrix A, Matrix B, Matrix C)
{
    for (int i = 0; i < C.rows; i += TILE)
        for (int j = 0; j < C.cols; j += TILE)
            ComputeTile(A, B, C, i, j);
}
