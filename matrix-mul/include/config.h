#ifndef CONFIG_H
#define CONFIG_H
#include "benchmark.h"
#include "timer.h"

constexpr int N = 1;       // Test case loop
constexpr int SIZE = 1024; // Rows, cols of each matrix, it must be times of 8

#define PrintResult(name, avgTime, avgCycle) \
        printf("%-20s%-20lf%-20lf\n", (name), (avgTime), (avgCycle))

#endif