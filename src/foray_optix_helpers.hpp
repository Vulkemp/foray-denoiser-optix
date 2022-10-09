#pragma once

#include <optix_types.h>
#include <foray_exception.hpp>
#include <cuda_runtime.h>

namespace foray::optix
{
    inline void AssertCudaResult(CUresult result)
    {
        if (result != CUresult::CUDA_SUCCESS)
        {
            FORAY_THROWFMT("CUDA result assertion failed: \"{}\"", (int32_t)result)
        }
    }

    inline void AssertCudaResult(cudaError_t result)
    {
        if (result != cudaError_t::cudaSuccess)
        {
            FORAY_THROWFMT("CUDA result assertion failed: \"{}\"", (int32_t)result)
        }
    }

    inline void AssertOptiXResult(OptixResult result)
    {
        if (result != OptixResult::OPTIX_SUCCESS)
        {
            FORAY_THROWFMT("OptiX result assertion failed: \"{}\"", (int32_t)result)
        }
    }
} // namespace foray::optix
