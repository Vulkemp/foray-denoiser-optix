#pragma once
#include "foray_cuda.hpp"
#include <core/foray_managedbuffer.hpp>

namespace foray::optix {

    /// @brief Image data between Cuda and Vulkan is passed via an intermediary buffer
    struct CudaBuffer
    {
        core::ManagedBuffer Buffer;
#ifdef WIN32
        HANDLE Handle = {INVALID_HANDLE_VALUE};
#else
        int Handle = -1;
#endif
        void* CudaPtr = nullptr;

        void Create(core::Context* context, VkDeviceSize size, std::string_view name);

        void SetupExportHandles(core::Context* context);
        void Destroy();
    };

}  // namespace foray::optix
