#pragma once

#ifdef __INTELLISENSE__

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#define CALL_KERNEL(cuda_func, ...) cuda_func

#else

#define CALL_KERNEL(cuda_func, ...) cuda_func<<<__VA_ARGS__>>>

#endif