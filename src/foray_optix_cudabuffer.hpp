#pragma once
#include "foray_cuda.hpp"
#include <core/foray_managedbuffer.hpp>

namespace foray::optix {

    /// @brief Image data between Cuda and Vulkan is passed via an intermediary buffer
    struct CudaBuffer
    {
        core::ManagedBuffer Buffer;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
        HANDLE Handle = {INVALID_HANDLE_VALUE};
#else
        int Handle = -1;
#endif
        void* CudaPtr = nullptr;

        VkExtent2D       Size;
        VkDeviceSize     PixelSize;
        OptixPixelFormat PixelFormat;

        void Create(core::Context* context, VkExtent2D size, VkDeviceSize pixelSize, OptixPixelFormat pixelFormat, std::string_view name);

        operator OptixImage2D();

        void SetupExportHandles(core::Context* context);
        void Destroy();
    };

}  // namespace foray::optix
