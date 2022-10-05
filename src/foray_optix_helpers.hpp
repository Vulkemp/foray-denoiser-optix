#pragma once

#include <optix_types.h>
#include <hsk_exception.hpp>

namespace foray::optix
{
    inline void AssertCudaResult(CUresult result)
    {
        if (result != CUresult::CUDA_SUCCESS)
        {
            HSK_THROWFMT("CUDA result assertion failed: \"{}\"", (int32_t)result)
        }
    }

    inline void AssertOptiXResult(OptixResult result)
    {
        if (result != OptixResult::OPTIX_SUCCESS)
        {
            HSK_THROWFMT("OptiX result assertion failed: \"{}\"", (int32_t)result)
        }
    }
} // namespace foray::optix
