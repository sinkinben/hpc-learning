#include "matrix.h"
void MatrixMul(Matrix A, Matrix B, Matrix C)
{
    int m = A.rows, n = A.cols, k = C.cols;
    for (int i = 0; i < m; ++i)
        for (int idx = 0; idx < n; ++idx)
            for (int j = 0; j < k; ++j)
                C.Get(i, j) += A.Get(i, idx) * B.Get(idx, j);
}
