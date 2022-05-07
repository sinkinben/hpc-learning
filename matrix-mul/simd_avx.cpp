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

    /* store the result of C[i, j] */
    alignas(32) static float cval;

    /* store a colume of B in a tile */
    alignas(32) static float bvals[TILE];
    
    /* AVX variables */
    __m256 a, b, c;
    for (int i = x; i < x + TILE; ++i)
    {
        for (int j = y; j < y + TILE; ++j)
        {
            for (int idx = 0; idx < TILE; ++idx)
                bvals[idx] = B.Get(y + idx, j);

            a = _mm256_load_ps(&A.Get(i, y));
            b = _mm256_load_ps(bvals);
            c = _mm256_mul_ps(a, b);
            _mm256_store_ps(&cval, c);
            C.Get(i, j) = cval;
        }
    }
}

void MatrixMul(Matrix A, Matrix B, Matrix C)
{
    /* We use TILE * 32 bit AVX here. */
    assert(TILE == 8);
    for (int i = 0; i < C.rows; i += TILE)
        for (int j = 0; j < C.cols; j += TILE)
            ComputeTile(A, B, C, i, j);
}