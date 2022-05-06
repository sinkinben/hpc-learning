## A Modern Multi-Core Processor

**P1 - 实现 Parallelism 的 3 种不同方式**

- CPU 架构 (L1-Cache = L1 D + L1 I)
  - 超标量处理器 (Superscalar Processor) 与多核 (Multi-core)
- SIMD (Single Instruction Multiple Data) - 一个指令，并行执行多个数据流
- SPMD (Single Program Multiple Data) - 一个程序，并行执行多个数据流
- AVX (Advanced Vector Extensions) 指令集
  - 实现 SIMD 的指令集之一，类似的还有 SSE
- 两类 ALU: Scalar ALU 和 Vector ALU

**P2 - Memory Access**

CPU 并行演化 (slide-105 开始)：

- Running code on a simple processor
- Superscalar core





**SIMD/SIMT/SPMD**

[What is the diffrence between SPMD and SIMD?](https://stackoverflow.com/questions/5014293/what-is-the-diffrence-between-spmd-and-simd)

SIMD is vectorization at the instruction level - each CPU instruction processes multiple data elements.

```cpp
for (i = 1 to INT_MAX)
{
	a += i;
	b += i;
}
```

The code above can be "SIMD" optimization.

SPMD is a much higher level abstraction where processes or programs are split across multiple processors and operate on different subsets of the data.