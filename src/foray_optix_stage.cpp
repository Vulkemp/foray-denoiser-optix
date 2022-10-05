#include "foray_optix_stage.hpp"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include "foray_optix_helpers.hpp"
#include <cuda_runtime.h>

using namespace hsk;

namespace foray::optix
{
    void OptixDebugCallback(unsigned int level, const char *tag, const char *message, void *cbdata)
    {
        spdlog::level::level_enum loglevel;
        switch (level)
        {
        case 1:
            loglevel = spdlog::level::level_enum::critical;
            break;
        case 2:
            loglevel = spdlog::level::level_enum::err;
            break;
        case 3:
            loglevel = spdlog::level::level_enum::warn;
            break;
        case 4:
        default:
            loglevel = spdlog::level::level_enum::info;
            break;
        }

        hsk::logger()->log(loglevel, "[OptiX::{}] {}", tag, message);
    }

    void OptiXDenoiserStage::Init(hsk::ManagedImage *noisy, hsk::ManagedImage *baseColor, hsk::ManagedImage *normal, bool useTemporal)
    {
        AssertCudaResult(cuInit(0)); // Initialize CUDA driver API.

        CUdevice device = 0;
        AssertCudaResult(cuCtxCreate(&mCudaContext, CU_CTX_SCHED_SPIN, device));

        // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
        AssertCudaResult(cuStreamCreate(&mCudaStream, CU_STREAM_DEFAULT));

        AssertOptiXResult(optixInit());
        AssertOptiXResult(optixDeviceContextCreate(mCudaContext, nullptr, &mOptixDevice));
        AssertOptiXResult(optixDeviceContextSetLogCallback(mOptixDevice, &OptixDebugCallback, nullptr, 4));

        mDenoiserOptions = OptixDenoiserOptions{.guideAlbedo = 1, .guideNormal = 1};
        OptixDenoiserModelKind modelKind = OPTIX_DENOISER_MODEL_KIND_HDR;
        AssertOptiXResult(optixDenoiserCreate(mOptixDevice, modelKind, &mDenoiserOptions, &mOptixDenoiser));
    }

    void OptiXDenoiserStage::RecordFrame(hsk::FrameRenderInfo &renderInfo) {}
    void OptiXDenoiserStage::CreateFixedSizeComponents() {}
    void OptiXDenoiserStage::DestroyFixedComponents() {}
    void OptiXDenoiserStage::CreateResolutionDependentComponents()
    {
        VkExtent3D extent = {mContext->Swapchain.extent.width, mContext->Swapchain.extent.height, 1};

        VkDeviceSize size = (VkDeviceSize)extent.width * (VkDeviceSize)extent.height * 4 * sizeof(fp32_t);

        // Using direct method
        VkBufferUsageFlags usage{VkBufferUsageFlagBits::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT};

        hsk::ManagedBuffer::ManagedBufferCreateInfo bufCi(usage, size, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, "OptiX Denoise Noisy Input");

        mInputBuffers[0].Buffer.Create(mContext, bufCi);
        mInputBuffers[0].Setup(mContext);

        bufCi.Name = "OptiX Denoise Position Input";
        mInputBuffers[1].Buffer.Create(mContext, bufCi);
        mInputBuffers[1].Setup(mContext);

        bufCi.Name = "OptiX Denoise Normal Input";
        mInputBuffers[2].Buffer.Create(mContext, bufCi);
        mInputBuffers[2].Setup(mContext);

        // Output image/buffer

        bufCi.Name = "OptiX Denoise Output";
        mOutputBuffer.Buffer.Create(mContext, bufCi);
        mOutputBuffer.Setup(mContext);

        // Computing the amount of memory needed to do the denoiser
        AssertOptiXResult(optixDenoiserComputeMemoryResources(mOptixDenoiser, extent.width, extent.height, &mDenoiserSizes));

        AssertCudaResult(cudaMalloc((void **)&mCudaStateBuffer, mDenoiserSizes.stateSizeInBytes));
        AssertCudaResult(cudaMalloc((void **)&mCudaScratchBuffer, mDenoiserSizes.withoutOverlapScratchSizeInBytes));
        AssertCudaResult(cudaMalloc((void **)&mCudaMinRGB, 4 * sizeof(float)));

        AssertOptiXResult(optixDenoiserSetup(mOptixDenoiser, mCudaStream, extent.width, extent.height, mCudaStateBuffer,
                                            mDenoiserSizes.stateSizeInBytes, mCudaScratchBuffer, mDenoiserSizes.withoutOverlapScratchSizeInBytes));
    }
    void OptiXDenoiserStage::DestroyResolutionDependentComponents()
    {
        mInputBuffers[0].Destroy();
        mInputBuffers[1].Destroy();
        mInputBuffers[2].Destroy();
        mOutputBuffer.Destroy();

        if (!!mCudaStateBuffer)
        {
            AssertCudaResult(cudaFree(reinterpret_cast<void *>(mCudaStateBuffer)));
        }
        if (!!mCudaScratchBuffer)
        {
            AssertCudaResult(cudaFree(reinterpret_cast<void *>(mCudaScratchBuffer)));
        }
        if (!!mCudaMinRGB)
        {
            AssertCudaResult(cudaFree(reinterpret_cast<void *>(mCudaMinRGB)));
        }
    }

    void OptiXDenoiserStage::CudaBuffer::Setup(const VkContext *context)
    {
        VkMemoryGetFdInfoKHR memInfo{
            .sType = VkStructureType::VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
            .memory = Buffer.GetAllocationInfo().deviceMemory,
            .handleType = VkExternalMemoryHandleTypeFlagBits::VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
        context->DispatchTable.getMemoryFdKHR(&memInfo, &Handle);
#ifdef WIN32
#pragma message "Windows Support WIP"

        // buf.handle = m_device.getMemoryWin32HandleKHR({memInfo.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32});
#else
#endif
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(context->Device, Buffer.GetBuffer(), &requirements);

        cudaExternalMemoryHandleDesc cudaExtMemHandleDesc{};
        cudaExtMemHandleDesc.size = requirements.size;
#ifdef WIN32
        // cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        // cudaExtMemHandleDesc.handle.win32.handle = buf.handle;
#else
        cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
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
        cudaExtBufferDesc.size = requirements.size;
        cudaExtBufferDesc.flags = 0;
        AssertCudaResult(cudaExternalMemoryGetMappedBuffer(&CudaPtr, cudaExtMemVertexBuffer, &cudaExtBufferDesc));
    }

    void OptiXDenoiserStage::CudaBuffer::Destroy()
    {
        Buffer.Destroy();
#ifdef WIN32
        CloseHandle(Handle);
#else
        if (Handle != -1)
        {
            close(Handle);
            Handle = -1;
        }
#endif
    }

} // namespace foray::optix
