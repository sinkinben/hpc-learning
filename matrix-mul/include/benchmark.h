#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__
#include <cstdint>
#include <chrono>
class Benchmark
{
public:
    static inline uint64_t GetCPUCycle();
};

inline uint64_t Benchmark::GetCPUCycle()
{
#ifdef __x86_64__
    unsigned int lo, hi;
    __asm__ __volatile__("lfence"
                         :
                         :
                         : "memory");
    __asm__ __volatile__("rdtsc"
                         : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#elif __aarch64__
    uint64_t v = 0;
    asm volatile("isb"
                 :
                 :
                 : "memory");
    asm volatile("mrs %0, cntvct_el0"
                 : "=r"(v));
    return v;
#else
    printf("unknown arch\n");
    return 0;
#endif
}
#endif