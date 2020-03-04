#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <windows.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "helper_cudaVS.h"

cudaError_t shflReduction(thrust::host_vector<double>& input);
cudaError_t thrustReduction(thrust::host_vector<double>& input);
cudaError_t cublasAbsReduction(thrust::host_vector<double>& input);


constexpr int N = 256 * 256 * 256;
constexpr int BLOCK_DIM = 1024;
constexpr int GRID_DIM = (N + BLOCK_DIM - 1) / BLOCK_DIM;
constexpr int WARP_DIM = ((N % BLOCK_DIM) + 31) / 32; // the number of warps to be active
constexpr unsigned FULL_MASK = 0xffffffff;

class Timer
{
public:
    Timer()
    {
        QueryPerformanceFrequency(&_freq);
    }

    void start()
    {
        QueryPerformanceCounter(&_start);
    }
    void end()
    {
        QueryPerformanceCounter(&_end);
    }

    double durationMS()
    {
        double time = static_cast<double>(_end.QuadPart - _start.QuadPart) * 1000.0 / _freq.QuadPart;
        return time;
    }
private:
    LARGE_INTEGER _freq;
    LARGE_INTEGER _start, _end;
};


__global__
void shflReduce_k(double* g_idata, double* g_odata)
{
    // shared memory for each warp sum
    __shared__ double smem[32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory size to be written
    int smem_size = (blockIdx.x < N / BLOCK_DIM) ? 32 : WARP_DIM;

    // calculate lane index and warp index
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;

    // calculate which thread will participate shfl operation
    unsigned mask = __ballot_sync(FULL_MASK, idx < N);
    double val;
    if (idx < N)
    {
        val = g_idata[idx];
        for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_xor_sync(mask, val, offset);
        }
    }

    // save warp sum to shared memory
    if (laneIdx == 0)
    {
        smem[warpIdx] = val;
    }

    // block synchronization
    __syncthreads();

    // last warp reduce
    mask = __ballot_sync(FULL_MASK, threadIdx.x < smem_size);
    if (threadIdx.x < smem_size)
    {
        val = smem[laneIdx];
        for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_xor_sync(mask, val, offset);
        }
    }

    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = val;
    }
}

int main()
{
    thrust::host_vector<double> h_input;

    for (int i = 0; i < N; ++i)
    {
        h_input.push_back(i);
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    // summation with shuffle reduction
    cudaStatus = shflReduction(h_input);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "shflReduction failed!");
        return 1;
    }

    // summation with thrust reduction
    cudaStatus = thrustReduction(h_input);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "thrustReduction failed!");
        return 1;
    }

    // summation with cublas absolute reduction;
    cudaStatus = cublasAbsReduction(h_input);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cublasAbsReduction failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t shflReduction(thrust::host_vector<double> &input)
{
    cudaError_t cudaStatus;
    Timer timer;

    thrust::device_vector<double> d_input;
    thrust::device_vector<double> d_output;

    thrust::host_vector<double> h_output;

    d_input.resize(input.size());
    d_output.resize(GRID_DIM);

    h_output.resize(GRID_DIM);

    d_input = input;

    timer.start();
    // Launch a kernel on the GPU with one thread for each element.
    CALL_KERNEL(shflReduce_k, GRID_DIM, BLOCK_DIM)(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()));

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    h_output = d_output;

    double sum = 0.0f;
    for (int i = 0; i < GRID_DIM; ++i)
    {
        sum += h_output[i];
    }
    
    timer.end();

    // result
    printf("*** Shuffle Reduction ***\n");
    printf("sum          : %f\n", sum);
    printf("time         : %.6lf ms\n", timer.durationMS());
    printf("grids        : <<<%d, %d>>>\n", GRID_DIM, BLOCK_DIM);

    return cudaStatus;
}

cudaError_t thrustReduction(thrust::host_vector<double>& input)
{
    cudaError_t cudaStatus;
    Timer timer;

    thrust::device_vector<double> d_input;

    d_input.resize(input.size());
    d_input = input;

    timer.start();

    // Launch a kernel on the GPU.
    double sum = thrust::reduce(d_input.begin(), d_input.end());

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    timer.end();

    // result
    printf("*** Thrust Reduction ***\n");
    printf("sum          : %f\n", sum);
    printf("time         : %.6lf ms\n", timer.durationMS());

    return cudaStatus;
}

cudaError_t cublasAbsReduction(thrust::host_vector<double>& input)
{
    cudaError_t cudaStatus;
    Timer timer;
    int size = input.size();

    thrust::device_vector<double> d_input;

    d_input.resize(size);
    d_input = input;
    double sum;

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    timer.start();

    // Launch a kernel on the GPU.
    cublasDasum(cublasHandle, size, thrust::raw_pointer_cast(d_input.data()), 1, &sum);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    timer.end();

    // result
    printf("*** Cublas Abs Reduction ***\n");
    printf("sum          : %f\n", sum);
    printf("time         : %.6lf ms\n", timer.durationMS());

    return cudaStatus;
}
