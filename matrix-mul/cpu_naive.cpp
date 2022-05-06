#include "matrix.h"

void MatrixMul(Matrix A, Matrix B, Matrix C)
{
    int m = A.rows, n = A.cols, k = C.cols;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            float res = 0;
            for (int idx = 0; idx < n; ++idx)
                res += A.Get(i, idx) * B.Get(idx, j);
            C.Get(i, j) = res;
        }
    }
}
