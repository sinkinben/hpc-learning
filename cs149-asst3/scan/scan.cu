#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"
#include "debug.h"

#define THREADS_PER_BLOCK 256

__global__ void print(int *arr, int N)
{
    printf("CUDA arr\n");
    for (int i = 0; i < N; ++i)
        printf("%d ", arr[i]);
    printf("\n");
}

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upSweep(int *arr, int N, int d)
{
    int d2 = d * 2;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * d2;
    if (idx < N)
        arr[idx + d2 - 1] += arr[idx + d - 1];
}

__global__ void downSweep(int *arr, int N, int d)
{
    int d2 = d * 2;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * d2;
    if (idx < N)
    {
        int t = arr[idx + d - 1];
        arr[idx + d - 1] = arr[idx + d2 - 1];
        arr[idx + d2 - 1] += t;
    }
}

// exclusive_scan --
void exclusive_scan(int *input, int N, int *result)
{
    constexpr int block = 512;

    // up sweep
    for (int d = N / 2, offset = 1; d >= 1; d /= 2, offset *= 2)
    {
        int grid = d / block;
        if (grid > 0)
            upSweep<<<grid, block>>>(result, N, offset);
        else
            upSweep<<<1, d>>>(result, N, offset);
    }
    cudaCheckError(cudaDeviceSynchronize());

    int zero = 0;
    cudaMemcpy(&result[N - 1], &zero, sizeof(int), cudaMemcpyHostToDevice);

    // down sweep
    for (int d = 1, offset = N / 2; d < N; d *= 2, offset /= 2)
    {
        int grid = d / block;
        if (grid > 0)
            downSweep<<<grid, block>>>(result, N, offset);
        else
            downSweep<<<1, d>>>(result, N, offset);
    }
}

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int *inarray, int *end, int *resultarray)
{
    int *device_result;
    int *device_input;
    int N = end - inarray;

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);

    cudaCheckError(cudaMalloc((void **)&device_result, sizeof(int) * rounded_length));
    cudaCheckError(cudaMalloc((void **)&device_input, sizeof(int) * rounded_length));
    assert(device_input != nullptr && device_result != nullptr);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, rounded_length, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    cudaFree(device_input);
    cudaFree(device_result);
    return overallDuration;
}

// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int *inarray, int *end, int *resultarray)
{

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void adjacentEqual(int *input, int *output, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    output[idx] = (idx + 1 < N) && (input[idx] == input[idx + 1]);
}
__global__ void collectRepeats(int *prefix_sum, int *output, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx + 1 < N && (prefix_sum[idx + 1] - prefix_sum[idx] == 1))
    {
        output[prefix_sum[idx]] = max(output[prefix_sum[idx]], idx);
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int *device_input, int N, int *device_output, int *equal_res)
{
    int rounded = nextPow2(N);

    constexpr int block = 16;
    int grid = (N + block - 1) / block;

    adjacentEqual<<<grid, block>>>(device_input, equal_res, N);
    cudaCheckError(cudaDeviceSynchronize());

    exclusive_scan(nullptr, rounded, equal_res);
    cudaCheckError(cudaDeviceSynchronize());

    // print<<<1, 1>>>(equal_res, N);
    // cudaCheckError(cudaDeviceSynchronize());

    collectRepeats<<<grid, block>>>(equal_res, device_output, N);
    cudaCheckError(cudaDeviceSynchronize());

    // print<<<1, 1>>>(device_output, N);
    // cudaCheckError(cudaDeviceSynchronize());

    int cnt = 0;
    cudaCheckError(cudaMemcpy(&cnt, &equal_res[N - 1], sizeof(int), cudaMemcpyDeviceToHost));

    return cnt;
}

//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length)
{

    int *device_input;
    int *device_output;
    int *buffer;
    int rounded_length = nextPow2(length);

    cudaCheckError(cudaMalloc((void **)&device_input, rounded_length * sizeof(int)));
    cudaCheckError(cudaMalloc((void **)&device_output, rounded_length * sizeof(int)));
    cudaCheckError(cudaMalloc((void **)&buffer, rounded_length * sizeof(int)));
    cudaCheckError(cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    int result = find_repeats(device_input, length, device_output, buffer);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(buffer);

    float duration = endTime - startTime;
    return duration;
}

void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
