## SIMD

- MMX: Multimedia extension 
- SSE: Streaming SIMD extension 
- AVX: Advanced vector extensions





The 256 bit (32 bytes) registers have enough space to store:

- 8 single-precision floats (_ps, Packed Single-precision)
- 4 double-precision floats (_pd, Packed Double-precision)
- 32 8-bit integers ( _epi8 signed char, or _epu8 unsigned char)
- 16 16-bit integers (_epi16 signed short, or _epu16 unsigned short)
- 8 32-bit integers (_epi32, Packed signed Integer, or _epu32, Packed Unsigned integer)
- 4 64-bit integers (_epi64 signed long)





两种方式实现 SIMD 优化：https://zhuanlan.zhihu.com/p/55327037



AVX2 Data Types:

```cpp
__m256  f; // = {float f0, f1, ..., f7}
__m256d d; // = {double d0, d1, d2, d3}
__m128i i; // 32 8-bit, 16 16-bit, 8 32-bit, or 4 64-bit ints
```



**Example**

```cpp
#include <immintrin.h>
#include <x86intrin.h>
#include <cstdio>
int main()
{
    __attribute__((aligned(32))) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    __attribute__((aligned(32))) float b[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    __attribute__((aligned(32))) float c[8] = {0};

    printf("a = %p, b = %p, c = %p\n", a, b, c);

    __m256 A = _mm256_load_ps(a);
    __m256 B = _mm256_load_ps(b);
    __m256 C = _mm256_mul_ps(A, B);

    _mm256_store_ps(c, C);

    for (int i = 0; i < 8; ++i)
        printf("%f ", c[i]);
}
```



## Ref

- [Intel - Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX2&ig_expand=634)