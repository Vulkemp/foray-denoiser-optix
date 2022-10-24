#include "foray_optix_cudabuffer.hpp"
#include "foray_optix_helpers.hpp"

namespace foray::optix {
    void CudaBuffer::SetupExportHandles(core::Context* context)
    {
#ifdef WIN32
        VkMemoryGetWin32HandleInfoKHR memInfo{
            .sType      = VkStructureType::VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
            .memory     = Buffer.GetAllocationInfo().deviceMemory,
            .handleType = VkExternalMemoryHandleTypeFlagBits::VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
        };

        AssertVkResult(context->VkbDispatchTable->getMemoryWin32HandleKHR(&memInfo, &Handle));

#else
        VkMemoryGetFdInfoKHR memInfo{.sType      = VkStructureType::VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                                     .memory     = Buffer.GetAllocationInfo().deviceMemory,
                                     .handleType = VkExternalMemoryHandleTypeFlagBits::VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
        context->VkbDispatchTable->getMemoryFdKHR(&memInfo, &Handle);
#endif
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(context->Device(), Buffer.GetBuffer(), &requirements);

        cudaExternalMemoryHandleDesc cudaExtMemHandleDesc{};
        cudaExtMemHandleDesc.size = requirements.size;
#ifdef WIN32
        cudaExtMemHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        cudaExtMemHandleDesc.handle.win32.handle = Handle;
#else
        cudaExtMemHandleDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
        cudaExtMemHandleDesc.handle.fd = Handle;
#endif

        cudaExternalMemory_t cudaExtMemVertexBuffer{};
        AssertCudaResult(cudaImportExternalMemory(&cudaExtMemVertexBuffer, &cudaExtMemHandleDesc));

#ifndef WIN32
        // fd got consumed
        cudaExtMemHandleDesc.handle.fd = -1;
#endif

        cudaExternalMemoryBufferDesc cudaExtBufferDesc{};
        cudaExtBufferDesc.offset = 0;
        cudaExtBufferDesc.size   = requirements.size;
        cudaExtBufferDesc.flags  = 0;
        AssertCudaResult(cudaExternalMemoryGetMappedBuffer(&CudaPtr, cudaExtMemVertexBuffer, &cudaExtBufferDesc));
    }

    void CudaBuffer::Destroy()
    {
        Buffer.Destroy();
#ifdef WIN32
        if(Handle != INVALID_HANDLE_VALUE)
        {
            CloseHandle(Handle);
            Handle = INVALID_HANDLE_VALUE;
        }
#else
        if(Handle != -1)
        {
            close(Handle);
            Handle = -1;
        }
#endif
    }

}  // namespace foray::optix
