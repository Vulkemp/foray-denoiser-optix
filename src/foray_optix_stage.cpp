#include "foray_optix_stage.hpp"
#include "foray_optix_helpers.hpp"
#include <core/foray_vkcontext.hpp>
#include <cuda_runtime.h>
#include <foray_logger.hpp>
#include <foray_vulkan.hpp>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

namespace foray::optix {
    void OptixDebugCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
    {
        spdlog::level::level_enum loglevel;
        switch(level)
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

        logger()->log(loglevel, "[OptiX::{}] {}", tag, message);
    }

    void OptiXDenoiserStage::Init(const core::VkContext* context, const stages::DenoiserConfig& config)
    {

        Destroy();
        mContext       = context;
        mPrimaryInput  = config.PrimaryInput;
        mAlbedoInput   = config.AlbedoInput;
        mNormalInput   = config.NormalInput;
        mPrimaryOutput = config.PrimaryOutput;
        mSemaphore     = config.Semaphore;

        CreateFixedSizeComponents();
        CreateResolutionDependentComponents();
    }

    void OptiXDenoiserStage::BeforeDenoise(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo)
    {
        {  // STEP #1    Memory barriers before transfer
            VkImageMemoryBarrier rtImgMemBarrier{
                .sType               = VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask       = VkAccessFlagBits::VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask       = VkAccessFlagBits::VK_ACCESS_TRANSFER_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image               = mPrimaryInput->GetImage(),
                .subresourceRange =
                    VkImageSubresourceRange{
                        .aspectMask     = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel   = 0U,
                        .levelCount     = 1U,
                        .baseArrayLayer = 0U,
                        .layerCount     = 1U,
                    },
            };
            renderInfo.GetImageLayoutCache().Set(mPrimaryInput, rtImgMemBarrier, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkImageMemoryBarrier gbufImgMemBarrier{
                .sType               = VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask       = VkAccessFlagBits::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dstAccessMask       = VkAccessFlagBits::VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout           = VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout           = VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .subresourceRange =
                    VkImageSubresourceRange{
                        .aspectMask     = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel   = 0U,
                        .levelCount     = 1U,
                        .baseArrayLayer = 0U,
                        .layerCount     = 1U,
                    },
            };

            std::vector<VkImageMemoryBarrier> barriers;
            barriers.reserve(2);

            barriers.push_back(rtImgMemBarrier);

            vkCmdPipelineBarrier(cmdBuffer, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VkDependencyFlagBits::VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(), barriers.data());

            barriers.clear();

            renderInfo.GetImageLayoutCache().Set(mAlbedoInput, gbufImgMemBarrier, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            barriers.push_back(gbufImgMemBarrier);

            renderInfo.GetImageLayoutCache().Set(mNormalInput, gbufImgMemBarrier, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            barriers.push_back(gbufImgMemBarrier);

            vkCmdPipelineBarrier(cmdBuffer, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VkDependencyFlagBits::VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(), barriers.data());
        }
        {  // STEP #2    Copy images to buffer
            VkBufferImageCopy imgCopy{
                .bufferOffset      = 0,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = VkImageSubresourceLayers{.aspectMask = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .imageOffset       = VkOffset3D{},
                .imageExtent       = VkExtent3D{.width = mContext->Swapchain.extent.width, .height = mContext->Swapchain.extent.height, .depth = 1},
            };

            vkCmdCopyImageToBuffer(cmdBuffer, mPrimaryInput->GetImage(), VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mInputBuffers[0].Buffer.GetBuffer(), 1, &imgCopy);
            vkCmdCopyImageToBuffer(cmdBuffer, mAlbedoInput->GetImage(), VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mInputBuffers[1].Buffer.GetBuffer(), 1, &imgCopy);
            vkCmdCopyImageToBuffer(cmdBuffer, mNormalInput->GetImage(), VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mInputBuffers[2].Buffer.GetBuffer(), 1, &imgCopy);
        }
        {  // STEP #3    Memory barriers after transfer
            VkImageMemoryBarrier rtImgMemBarrier{
                .sType               = VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask       = VkAccessFlagBits::VK_ACCESS_TRANSFER_READ_BIT,
                .dstAccessMask       = VkAccessFlagBits::VK_ACCESS_NONE,
                .oldLayout           = VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout           = VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image               = mPrimaryInput->GetImage(),
                .subresourceRange =
                    VkImageSubresourceRange{
                        .aspectMask     = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel   = 0U,
                        .levelCount     = 1U,
                        .baseArrayLayer = 0U,
                        .layerCount     = 1U,
                    },
            };
            VkImageMemoryBarrier gbufImgMemBarrier{
                .sType               = VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask       = VkAccessFlagBits::VK_ACCESS_TRANSFER_READ_BIT,
                .dstAccessMask       = VkAccessFlagBits::VK_ACCESS_NONE,
                .oldLayout           = VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout           = VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .subresourceRange =
                    VkImageSubresourceRange{
                        .aspectMask     = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel   = 0U,
                        .levelCount     = 1U,
                        .baseArrayLayer = 0U,
                        .layerCount     = 1U,
                    },
            };
            std::vector<VkImageMemoryBarrier> barriers;
            barriers.reserve(3);
            barriers.push_back(rtImgMemBarrier);
            gbufImgMemBarrier.image = mAlbedoInput->GetImage();
            barriers.push_back(gbufImgMemBarrier);
            gbufImgMemBarrier.image = mNormalInput->GetImage();
            barriers.push_back(gbufImgMemBarrier);
            vkCmdPipelineBarrier(cmdBuffer, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VkDependencyFlagBits::VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(), barriers.data());
        }
    }
    void OptiXDenoiserStage::AfterDenoise(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo)
    {
        {  // STEP #1    Memory barriers before transfer
            VkImageMemoryBarrier imgMemBarrier{
                .sType               = VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask       = VkAccessFlagBits::VK_ACCESS_NONE,
                .dstAccessMask       = VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image               = mPrimaryOutput->GetImage(),
                .subresourceRange =
                    VkImageSubresourceRange{
                        .aspectMask     = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel   = 0U,
                        .levelCount     = 1U,
                        .baseArrayLayer = 0U,
                        .layerCount     = 1U,
                    },
            };
            renderInfo.GetImageLayoutCache().Set(mPrimaryOutput, imgMemBarrier, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            vkCmdPipelineBarrier(cmdBuffer, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VkDependencyFlagBits::VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &imgMemBarrier);
        }
        {  // STEP #2    Copy buffer to image
            VkBufferImageCopy imgCopy{
                .bufferOffset      = 0,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = VkImageSubresourceLayers{.aspectMask = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .imageOffset       = VkOffset3D{},
                .imageExtent       = VkExtent3D{.width = mContext->Swapchain.extent.width, .height = mContext->Swapchain.extent.height, .depth = 1},
            };

            vkCmdCopyBufferToImage(cmdBuffer, mOutputBuffer.Buffer.GetBuffer(), mPrimaryOutput->GetImage(), VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imgCopy);
        }
        // {  // STEP #3    Memory barriers after transfer
        //     VkImageMemoryBarrier imgMemBarrier{
        //         .sType               = VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        //         .srcAccessMask       = VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT,
        //         .dstAccessMask       = VkAccessFlagBits::VK_ACCESS_TRANSFER_READ_BIT,
        //         .oldLayout           = VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        //         .newLayout           = VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        //         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        //         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        //         .image               = mPrimaryOutput->GetImage(),
        //         .subresourceRange =
        //             VkImageSubresourceRange{
        //                 .aspectMask     = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT,
        //                 .baseMipLevel   = 0U,
        //                 .levelCount     = 1U,
        //                 .baseArrayLayer = 0U,
        //                 .layerCount     = 1U,
        //             },
        //     };
        //     vkCmdPipelineBarrier(cmdBuffer, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT,
        //                          VkDependencyFlagBits::VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &imgMemBarrier);
        // }
    }
    void OptiXDenoiserStage::DispatchDenoise(uint64_t timelineValueBefore, uint64_t timelineValueAfter)
    {
        try
        {
            VkExtent2D size = mContext->Swapchain.extent;

            OptixPixelFormat pixelFormat      = mPixelFormat;
            auto             sizeofPixel      = mSizeOfPixel;
            uint32_t         rowStrideInBytes = sizeofPixel * size.width;

            //std::vector<OptixImage2D> inputLayer;  // Order: RGB, Albedo, Normal

            // Create and set our OptiX layers

            OptixImage2D imageBase{
                .width = size.width, .height = size.height, .rowStrideInBytes = rowStrideInBytes, .pixelStrideInBytes = (uint32_t)sizeofPixel, .format = pixelFormat};

            OptixDenoiserLayer layerBase{
                .input          = imageBase,
                .previousOutput = OptixImage2D{},
                .output         = imageBase,
            };

            OptixDenoiserLayer primaryLayer = layerBase;
            primaryLayer.input.data         = (CUdeviceptr)mInputBuffers[0].CudaPtr;
            primaryLayer.output.data        = (CUdeviceptr)mOutputBuffer.CudaPtr;

            OptixDenoiserGuideLayer guideLayer{
                .albedo = imageBase, .normal = imageBase, .flow = OptixImage2D{}, .previousOutputInternalGuideLayer = OptixImage2D{}, .outputInternalGuideLayer = OptixImage2D{}};

            guideLayer.albedo.data = (CUdeviceptr)mInputBuffers[1].CudaPtr;
            guideLayer.normal.data = (CUdeviceptr)mInputBuffers[2].CudaPtr;

            // Wait from Vulkan (Copy to Buffer)
            cudaExternalSemaphoreWaitParams waitParams{};
            waitParams.flags              = 0;
            waitParams.params.fence.value = timelineValueBefore;
            cudaWaitExternalSemaphoresAsync(&mCudaSemaphore, &waitParams, 1, nullptr);

            if(!!mCudaIntensity)
            {
                AssertOptiXResult(optixDenoiserComputeIntensity(mOptixDenoiser, mCudaStream, &primaryLayer.input, mCudaIntensity, mCudaScratchBuffer,
                                                                mDenoiserSizes.withoutOverlapScratchSizeInBytes));
            }

            OptixDenoiserParams denoiserParams{};
            denoiserParams.denoiseAlpha = OptixDenoiserAlphaMode::OPTIX_DENOISER_ALPHA_MODE_COPY;
            denoiserParams.hdrIntensity = mCudaIntensity;
            denoiserParams.blendFactor  = 0.0f;  // Fully denoised


            // Execute the denoiser
            AssertOptiXResult(optixDenoiserInvoke(mOptixDenoiser, mCudaStream, &denoiserParams, mCudaStateBuffer, mDenoiserSizes.stateSizeInBytes, &guideLayer, &primaryLayer, 1, 0,
                                                  0, mCudaScratchBuffer, mDenoiserSizes.withoutOverlapScratchSizeInBytes));


            AssertCudaResult(cudaStreamSynchronize(mCudaStream));  // Making sure the denoiser is done

            cudaExternalSemaphoreSignalParams sigParams{};
            sigParams.flags              = 0;
            sigParams.params.fence.value = timelineValueAfter;
            cudaSignalExternalSemaphoresAsync(&mCudaSemaphore, &sigParams, 1, mCudaStream);
        }
        catch(const std::exception& e)
        {
            logger();
        }
    }

    void OptiXDenoiserStage::CreateFixedSizeComponents()
    {
        AssertCudaResult(cuInit(0));  // Initialize CUDA driver API.

        CUdevice device = 0;
        AssertCudaResult(cuCtxCreate(&mCudaContext, CU_CTX_SCHED_SPIN, device));

        // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
        AssertCudaResult(cuStreamCreate(&mCudaStream, CU_STREAM_DEFAULT));

        AssertOptiXResult(optixInit());
        AssertOptiXResult(optixDeviceContextCreate(mCudaContext, nullptr, &mOptixDevice));
        AssertOptiXResult(optixDeviceContextSetLogCallback(mOptixDevice, &OptixDebugCallback, nullptr, 4));

        mDenoiserOptions                 = OptixDenoiserOptions{.guideAlbedo = 1, .guideNormal = 1};
        OptixDenoiserModelKind modelKind = OPTIX_DENOISER_MODEL_KIND_HDR;
        AssertOptiXResult(optixDenoiserCreate(mOptixDevice, modelKind, &mDenoiserOptions, &mOptixDenoiser));

#ifdef WIN32
        auto handleType = VkExternalSemaphoreHandleTypeFlagBits::VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        auto handleType                       = VkExternalSemaphoreHandleTypeFlagBits::VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
        std::memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
        externalSemaphoreHandleDesc.flags = 0;
#ifdef WIN32
        externalSemaphoreHandleDesc.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        externalSemaphoreHandleDesc.handle.win32.handle = reinterpret_cast<void*>(mSemaphore->GetHandle());
#else
        externalSemaphoreHandleDesc.type      = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        externalSemaphoreHandleDesc.handle.fd = mSemaphore->GetHandle();
#endif

        AssertCudaResult(cudaImportExternalSemaphore(&mCudaSemaphore, &externalSemaphoreHandleDesc));
    }
    void OptiXDenoiserStage::DestroyFixedComponents()
    {
        if(!!mCudaSemaphore)
        {
            cudaDestroyExternalSemaphore(mCudaSemaphore);
            mCudaSemaphore = nullptr;
        }
        if(!!mOptixDenoiser)
        {
            optixDenoiserDestroy(mOptixDenoiser);
            mOptixDenoiser = nullptr;
        }
        if(!!mOptixDevice)
        {
            optixDeviceContextDestroy(mOptixDevice);
            mOptixDevice = nullptr;
        }

        if(!!mCudaStream)
        {
            cuStreamDestroy(mCudaStream);
            mCudaStream = nullptr;
        }
        if(!!mCudaContext)
        {
            cuCtxDestroy(mCudaContext);
            mCudaContext = nullptr;
        }
    }
    void OptiXDenoiserStage::CreateResolutionDependentComponents()
    {
        VkExtent3D extent = {mContext->Swapchain.extent.width, mContext->Swapchain.extent.height, 1};

        VkDeviceSize size = (VkDeviceSize)extent.width * (VkDeviceSize)extent.height * mSizeOfPixel;

        // Using direct method
        VkBufferUsageFlags usage{VkBufferUsageFlagBits::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                 | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT};

        core::ManagedBuffer::ManagedBufferCreateInfo bufCi(usage, size, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                                                           VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, "OptiX Denoise Noisy Input");

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

        AssertCudaResult(cudaMalloc((void**)&mCudaStateBuffer, mDenoiserSizes.stateSizeInBytes));
        AssertCudaResult(cudaMalloc((void**)&mCudaScratchBuffer, mDenoiserSizes.withoutOverlapScratchSizeInBytes));
        AssertCudaResult(cudaMalloc((void**)&mCudaMinRGB, 4 * sizeof(float)));

        AssertOptiXResult(optixDenoiserSetup(mOptixDenoiser, mCudaStream, extent.width, extent.height, mCudaStateBuffer, mDenoiserSizes.stateSizeInBytes, mCudaScratchBuffer,
                                             mDenoiserSizes.withoutOverlapScratchSizeInBytes));
    }
    void OptiXDenoiserStage::DestroyResolutionDependentComponents()
    {
        mInputBuffers[0].Destroy();
        mInputBuffers[1].Destroy();
        mInputBuffers[2].Destroy();
        mOutputBuffer.Destroy();

        if(!!mCudaStateBuffer)
        {
            AssertCudaResult(cudaFree(reinterpret_cast<void*>(mCudaStateBuffer)));
        }
        if(!!mCudaScratchBuffer)
        {
            AssertCudaResult(cudaFree(reinterpret_cast<void*>(mCudaScratchBuffer)));
        }
        if(!!mCudaMinRGB)
        {
            AssertCudaResult(cudaFree(reinterpret_cast<void*>(mCudaMinRGB)));
        }
    }

    void OptiXDenoiserStage::CudaBuffer::Setup(const core::VkContext* context)
    {
#ifdef WIN32
        VkMemoryGetWin32HandleInfoKHR memInfo{
            .sType      = VkStructureType::VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
            .memory     = Buffer.GetAllocationInfo().deviceMemory,
            .handleType = VkExternalMemoryHandleTypeFlagBits::VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
        };

        VkDevice device = context->Device;

        PFN_vkGetMemoryWin32HandleKHR getHandleFunc = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));

        if(!getHandleFunc)
        {
            Exception::Throw("Unable to resolve vkGetMemoryWin32HandleKHR device proc addr!");
        }

        getHandleFunc(device, &memInfo, &Handle);

#else
        VkMemoryGetFdInfoKHR memInfo{.sType      = VkStructureType::VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                                     .memory     = Buffer.GetAllocationInfo().deviceMemory,
                                     .handleType = VkExternalMemoryHandleTypeFlagBits::VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
        context->DispatchTable.getMemoryFdKHR(&memInfo, &Handle);
#endif
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(context->Device, Buffer.GetBuffer(), &requirements);

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

    void OptiXDenoiserStage::CudaBuffer::Destroy()
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
