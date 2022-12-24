#include "foray_optix_cudabuffer.hpp"
#include "foray_optix_helpers.hpp"

namespace foray::optix {
    void CudaBuffer::Create(core::Context* context, VkExtent2D size, VkDeviceSize pixelSize, OptixPixelFormat pixelFormat, std::string_view name)
    {
        Size = size;
        PixelSize = pixelSize;
        PixelFormat = pixelFormat;

        VkDeviceSize bufferSize = size.width * size.height * pixelSize;

        VkBufferUsageFlags usage{VkBufferUsageFlagBits::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                 | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT};

        core::ManagedBuffer::CreateInfo bufCi(usage, bufferSize, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                                           VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, name);

        VkExternalMemoryBufferCreateInfo extMemBufCi{.sType = VkStructureType::VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
        extMemBufCi.handleTypes = VkExternalMemoryHandleTypeFlagBits::VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        extMemBufCi.handleTypes = VkExternalMemoryHandleTypeFlagBits::VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        bufCi.BufferCreateInfo.pNext = &extMemBufCi;

        Buffer.Create(context, bufCi);
        SetupExportHandles(context);
    }

    void CudaBuffer::SetupExportHandles(core::Context* context)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
        VkMemoryGetWin32HandleInfoKHR memInfo{
            .sType      = VkStructureType::VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
            .memory     = Buffer.GetAllocationInfo().deviceMemory,
            .handleType = VkExternalMemoryHandleTypeFlagBits::VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
        };

        PFN_vkGetMemoryWin32HandleKHR getFunc = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(context->Device(), "vkGetMemoryWin32HandleKHR");
        Assert(!!getFunc);
        HANDLE handle;
        AssertVkResult(getFunc(context->Device(), & memInfo, &handle));
        Handle = handle;
        Assert(Handle != 0 && Handle != INVALID_HANDLE_VALUE, "getMemoryWin32HandleKHR returned VK_SUCCESS, but returned handle is invalid");

#else
        VkMemoryGetFdInfoKHR memInfo{.sType      = VkStructureType::VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                                     .memory     = Buffer.GetAllocationInfo().deviceMemory,
                                     .handleType = VkExternalMemoryHandleTypeFlagBits::VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
        context->VkbDispatchTable->getMemoryFdKHR(&memInfo, &Handle);
#endif

        cudaExternalMemoryHandleDesc cudaExtMemHandleDesc{};
        cudaExtMemHandleDesc.size = Buffer.GetSize();
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
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
        cudaExtBufferDesc.size   = Buffer.GetSize();
        cudaExtBufferDesc.flags  = 0;
        AssertCudaResult(cudaExternalMemoryGetMappedBuffer(&CudaPtr, cudaExtMemVertexBuffer, &cudaExtBufferDesc));
    }

    CudaBuffer::operator OptixImage2D()
    {
        return OptixImage2D
        {
            /// Pointer to the actual pixel data.
            .data = reinterpret_cast<CUdeviceptr>(CudaPtr),
            /// Width of the image (in pixels)
            .width = Size.width,
            /// Height of the image (in pixels)
            .height = Size.height,
            /// Stride between subsequent rows of the image (in bytes).
            .rowStrideInBytes = Size.width * (uint32_t)PixelSize,
            /// Stride between subsequent pixels of the image (in bytes).
            /// If set to 0, dense packing (no gaps) is assumed.
            /// For pixel format OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER it must be set to
            /// at least OptixDenoiserSizes::internalGuideLayerSizeInBytes.
            .pixelStrideInBytes = (uint32_t)PixelSize,
            /// Pixel format.
            .format = PixelFormat
        };
    }

    void CudaBuffer::Destroy()
    {
        Buffer.Destroy();
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
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
