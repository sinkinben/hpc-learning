#define Get(array, i, j, cols) (array[i * cols + j])
#define TILE (2)
/* A = (m x n), B = (n x k), C = (m x k) */
static inline void ComputeTile(
    uniform int m, uniform int n, uniform int k, 
    uniform float A[], 
    uniform float B[], 
    uniform float C[],
    uniform int tileRow, uniform int tileCol
)
{
    for (uniform int i = tileRow; i < tileRow + TILE; ++i)
    {
        for (uniform int j = tileCol; j < tileCol + TILE; ++j)
        {
            for (uniform int idx = tileCol; idx < tileCol + TILE; ++idx)
            {
                Get(C, i, j, k) += Get(A, i, idx, n) * Get(B, idx, j, k);
            }
        }
    }
}

export void MatrixMul(
    uniform int m, uniform int n, uniform int k, 
    uniform float A[], 
    uniform float B[], 
    uniform float C[]
)
{
    uniform int tileRow;
    uniform int tileCol;
    for (tileRow = 0; tileRow < m; tileRow += TILE)
    {
        for (tileCol = 0; tileCol < k; tileCol += TILE)
        {
            ComputeTile(m, n, k, A, B, C, tileRow, tileCol);
        }
    }
}