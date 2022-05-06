/* Matrix multiplication based on AVX (Advanced Vector Extensions)
 * instructions. You need to add '-mavx' flag into gcc/g++ arguments.
 * However, it seems that AVX is not always good. Linus have criticized
 * it very fiercely.
 */
#include <immintrin.h>
#include <x86intrin.h>
#include "matrix.h"

constexpr int TILE = 8;
/* (x, y) is the index of one tile.
 * Please note that I don't check the bounder. 
 * The x + TILE, y + TILE may be out of index bounder.
 */
static inline void ComputeTile(Matrix A, Matrix B, Matrix C, int x, int y)
{
    int m = A.rows, n = A.cols, k = C.cols;
    alignas(32) float cval;
    for (int i = x; i < x + TILE; ++i)
    {
        for (int j = y; j < y + TILE; ++j)
        {
            __m256 a = _mm256_load_ps(&A.Get(i, y));
            __m256 b = _mm256_load_ps(&B.Get(j, y));
            __m256 c = _mm256_mul_ps(a, b);
            _mm256_store_ps(&cval, c);
            C.Get(i, j) = cval;
        }
    }
}

/* M.rows == M.cols */
void Transpose(Matrix M)
{
    assert(M.rows == M.cols);
    for (int i = 0; i < M.rows; ++i)
        for (int j = i + 1; j < M.cols; ++j)
            std::swap(M.Get(i, j), M.Get(j, i));
}

void MatrixMul(Matrix A, Matrix B, Matrix C)
{
    Transpose(B);
    for (int i = 0; i < C.rows; i += TILE)
        for (int j = 0; j < C.cols; j += TILE)
            ComputeTile(A, B, C, i, j);
}