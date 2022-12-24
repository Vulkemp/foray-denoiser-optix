#include "foray_optix_stage.hpp"
#include "foray_optix_helpers.hpp"
#include <bench/foray_devicebenchmark.hpp>
#include <core/foray_context.hpp>
#include <cuda_runtime.h>
#include <foray_logger.hpp>
#include <foray_vulkan.hpp>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

namespace foray::optix {

#pragma region Debug Callback

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

#pragma endregion
#pragma region Init

    void OptiXDenoiserStage::Init(core::Context* context, const stages::DenoiserConfig& config)
    {

        Destroy();
        mContext       = context;
        mPrimaryInput  = config.PrimaryInput;
        mAlbedoInput   = config.GBufferOutputs[(size_t)stages::GBufferStage::EOutput::Albedo];
        mNormalInput   = config.GBufferOutputs[(size_t)stages::GBufferStage::EOutput::Normal];
        mMotionInput   = config.GBufferOutputs[(size_t)stages::GBufferStage::EOutput::Motion];
        mPrimaryOutput = config.PrimaryOutput;
        mSemaphore     = config.Semaphore;
        mBenchmark     = config.Benchmark;

        Assert(!!mPrimaryInput, "Primary Input must be set");
        Assert(!!mPrimaryOutput, "Primary Output must be set");
        Assert(!!mSemaphore, "OptiX relies on device synchronized execution. A semaphore must be set");

        CreateFixedSizeComponents();
        CreateResolutionDependentComponents();
    }

    void OptiXDenoiserStage::CreateFixedSizeComponents()
    {
        if(!!mBenchmark)
        {
            std::vector<const char*> queryNames({bench::BenchmarkTimestamp::BEGIN, TIMESTAMP_COPYTOBUFFERS, TIMESTAMP_SCALEMOTION, TIMESTAMP_CUDA, bench::BenchmarkTimestamp::END});
            mBenchmark->Create(mContext, queryNames);
        }

        AssertCudaResult(cuInit(0));  // Initialize CUDA driver API.

        CUdevice device = 0;
        AssertCudaResult(cuCtxCreate(&mCudaContext, CU_CTX_SCHED_SPIN, device));

        // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
        AssertCudaResult(cuStreamCreate(&mCudaStream, CU_STREAM_DEFAULT));

        AssertOptiXResult(optixInit());
        AssertOptiXResult(optixDeviceContextCreate(mCudaContext, nullptr, &mOptixDevice));
        AssertOptiXResult(optixDeviceContextSetLogCallback(mOptixDevice, &OptixDebugCallback, nullptr, 4));

        mDenoiserOptions                 = OptixDenoiserOptions{.guideAlbedo = !!mAlbedoInput, .guideNormal = !!mNormalInput};
        OptixDenoiserModelKind modelKind = (!!mMotionInput) ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;
        AssertOptiXResult(optixDenoiserCreate(mOptixDevice, modelKind, &mDenoiserOptions, &mOptixDenoiser));

        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
        std::memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
        externalSemaphoreHandleDesc.flags = 0;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
        externalSemaphoreHandleDesc.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        externalSemaphoreHandleDesc.handle.win32.handle = reinterpret_cast<void*>(mSemaphore->GetHandle());
#else
        externalSemaphoreHandleDesc.type      = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        externalSemaphoreHandleDesc.handle.fd = mSemaphore->GetHandle();
#endif

        AssertCudaResult(cudaImportExternalSemaphore(&mCudaSemaphore, &externalSemaphoreHandleDesc));
    }

    void OptiXDenoiserStage::CreateResolutionDependentComponents()
    {
        VkExtent2D extent = mContext->GetSwapchainSize();

        VkDeviceSize sizeOfPixel = 4 * sizeof(uint16_t);

        mInputBuffers[EInputBufferKind::Source].Create(mContext, extent, sizeOfPixel, OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF4, "OptiX Denoise Noisy Input");

        if(!!mAlbedoInput)
        {
            mInputBuffers[EInputBufferKind::Albedo].Create(mContext, extent, sizeOfPixel, OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF4, "OptiX Denoise Albedo Input");
        }

        if(!!mNormalInput)
        {
            mInputBuffers[EInputBufferKind::Normal].Create(mContext, extent, sizeOfPixel, OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF4, "OptiX Denoise Normal Input");
        }

        if(!!mMotionInput)
        {
            VkDeviceSize sizeOfPixel = 2 * sizeof(float);
            mInputBuffers[EInputBufferKind::Motion].Create(mContext, extent, sizeOfPixel, OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT2, "OptiX Denoise Motion Input");
            mScaleMotionStage.Init(mContext, mMotionInput, &(mInputBuffers[EInputBufferKind::Motion].Buffer));
        }

        // Output image/buffer

        mOutputBuffer.Create(mContext, extent, sizeOfPixel, OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF4, "OptiX Denoise Output");

        // Computing the amount of memory needed to do the denoiser
        AssertOptiXResult(optixDenoiserComputeMemoryResources(mOptixDenoiser, extent.width, extent.height, &mDenoiserSizes));

        AssertCudaResult(cudaMalloc((void**)&mCudaStateBuffer, mDenoiserSizes.stateSizeInBytes));
        AssertCudaResult(cudaMalloc((void**)&mCudaScratchBuffer, mDenoiserSizes.withoutOverlapScratchSizeInBytes));
        AssertCudaResult(cudaMalloc((void**)&mCudaMinRGB, 4 * sizeof(float)));
        AssertCudaResult(cudaMalloc((void**)&mCudaIntensity, 1 * sizeof(float)));

        AssertOptiXResult(optixDenoiserSetup(mOptixDenoiser, mCudaStream, extent.width, extent.height, mCudaStateBuffer, mDenoiserSizes.stateSizeInBytes, mCudaScratchBuffer,
                                             mDenoiserSizes.withoutOverlapScratchSizeInBytes));
    }

#pragma endregion
#pragma region Render

    void OptiXDenoiserStage::BeforeDenoise(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo)
    {
        uint32_t frameIdx = renderInfo.GetFrameNumber();
        if(!!mBenchmark)
        {
            mBenchmark->CmdResetQuery(cmdBuffer, frameIdx);
            mBenchmark->CmdWriteTimestamp(cmdBuffer, frameIdx, bench::BenchmarkTimestamp::BEGIN, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT);
        }
        {  // STEP #1    Memory barriers before transfer
            core::ImageLayoutCache::Barrier2 barrier{.SrcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                     .SrcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                                     .DstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                     .DstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                                     .NewLayout     = VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};

            std::vector<VkImageMemoryBarrier2> barriers;
            barriers.reserve(3);

            barriers.push_back(renderInfo.GetImageLayoutCache().MakeBarrier(mPrimaryInput, barrier));
            if(!!mAlbedoInput)
            {
                barriers.push_back(renderInfo.GetImageLayoutCache().MakeBarrier(mAlbedoInput, barrier));
            }
            if(!!mNormalInput)
            {
                barriers.push_back(renderInfo.GetImageLayoutCache().MakeBarrier(mNormalInput, barrier));
            }

            VkDependencyInfo depInfo{
                .sType = VkStructureType::VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = (uint32_t)barriers.size(), .pImageMemoryBarriers = barriers.data()};

            vkCmdPipelineBarrier2(cmdBuffer, &depInfo);
        }
        {  // STEP #2    Copy images to buffer
            VkBufferImageCopy imgCopy{
                .bufferOffset      = 0,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = VkImageSubresourceLayers{.aspectMask = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .imageOffset       = VkOffset3D{},
                .imageExtent       = VkExtent3D{.width = mContext->GetSwapchainSize().width, .height = mContext->GetSwapchainSize().height, .depth = 1},
            };

            vkCmdCopyImageToBuffer(cmdBuffer, mPrimaryInput->GetImage(), VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   mInputBuffers[EInputBufferKind::Source].Buffer.GetBuffer(), 1, &imgCopy);
            if(!!mAlbedoInput)
            {
                vkCmdCopyImageToBuffer(cmdBuffer, mAlbedoInput->GetImage(), VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       mInputBuffers[EInputBufferKind::Albedo].Buffer.GetBuffer(), 1, &imgCopy);
            }
            if(!!mNormalInput)
            {
                vkCmdCopyImageToBuffer(cmdBuffer, mNormalInput->GetImage(), VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       mInputBuffers[EInputBufferKind::Normal].Buffer.GetBuffer(), 1, &imgCopy);
            }
        }
        if(!!mMotionInput)
        {
            if(!!mBenchmark)
            {
                mBenchmark->CmdWriteTimestamp(cmdBuffer, frameIdx, TIMESTAMP_COPYTOBUFFERS, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            }
            mScaleMotionStage.RecordFrame(cmdBuffer, renderInfo);
            if(!!mBenchmark)
            {
                mBenchmark->CmdWriteTimestamp(cmdBuffer, frameIdx, TIMESTAMP_SCALEMOTION, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
            }
        }
        else if(!!mBenchmark)
        {
            mBenchmark->CmdWriteTimestamp(cmdBuffer, frameIdx, TIMESTAMP_COPYTOBUFFERS, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
            mBenchmark->CmdWriteTimestamp(cmdBuffer, frameIdx, TIMESTAMP_SCALEMOTION, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        }
    }

    void OptiXDenoiserStage::DispatchDenoise(uint64_t timelineValueBefore, uint64_t timelineValueAfter)
    {
        try
        {
            VkExtent2D size = mContext->GetSwapchainSize();

            OptixDenoiserLayer primaryLayer{.input = mInputBuffers[EInputBufferKind::Source], .output = mOutputBuffer};

            if(!!mMotionInput)
            {
                // This is ok, because OptiX consumes previous output before writing the next
                primaryLayer.previousOutput = mOutputBuffer;
            }

            OptixDenoiserGuideLayer guideLayer{};

            if(!!mAlbedoInput)
            {
                guideLayer.albedo = mInputBuffers[EInputBufferKind::Albedo];
            }
            if(!!mNormalInput)
            {
                guideLayer.normal = mInputBuffers[EInputBufferKind::Normal];
            }
            if(!!mMotionInput)
            {
                guideLayer.flow = mInputBuffers[EInputBufferKind::Motion];
            }

            // Wait from Vulkan (Copy to Buffer)
            cudaExternalSemaphoreWaitParams waitParams{};
            waitParams.flags              = 0;
            waitParams.params.fence.value = timelineValueBefore;
            cudaWaitExternalSemaphoresAsync(&mCudaSemaphore, &waitParams, 1, nullptr);

            if(!!mCudaIntensity)
            {
                // Calculate intensity to improve performance in very bright or very dark scenes
                AssertOptiXResult(optixDenoiserComputeIntensity(mOptixDenoiser, mCudaStream, &primaryLayer.input, mCudaIntensity, mCudaScratchBuffer,
                                                                mDenoiserSizes.withoutOverlapScratchSizeInBytes));
            }

            OptixDenoiserParams denoiserParams{};
            denoiserParams.denoiseAlpha                  = OptixDenoiserAlphaMode::OPTIX_DENOISER_ALPHA_MODE_COPY;
            denoiserParams.hdrIntensity                  = mCudaIntensity;
            denoiserParams.blendFactor                   = 0.0f;  // Fully denoised
            denoiserParams.temporalModeUsePreviousLayers = mDenoisedFrames > 0;

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
        mDenoisedFrames++;
    }

    void OptiXDenoiserStage::AfterDenoise(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo)
    {
        uint32_t frameIdx = renderInfo.GetFrameNumber();
        if(!!mBenchmark)
        {
            mBenchmark->CmdWriteTimestamp(cmdBuffer, frameIdx, TIMESTAMP_CUDA, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        }
        {  // STEP #1    Memory barriers before transfer
            core::ImageLayoutCache::Barrier2 barrier{.SrcStageMask  = VK_PIPELINE_STAGE_2_NONE,
                                                     .SrcAccessMask = VK_ACCESS_2_NONE,
                                                     .DstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                     .DstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                     .NewLayout     = VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL};

            renderInfo.GetImageLayoutCache().CmdBarrier(cmdBuffer, mPrimaryOutput, barrier);
        }
        {  // STEP #2    Copy buffer to image
            VkBufferImageCopy imgCopy{
                .bufferOffset      = 0,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = VkImageSubresourceLayers{.aspectMask = VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .imageOffset       = VkOffset3D{},
                .imageExtent       = VkExtent3D{.width = mContext->GetSwapchainSize().width, .height = mContext->GetSwapchainSize().height, .depth = 1},
            };

            vkCmdCopyBufferToImage(cmdBuffer, mOutputBuffer.Buffer.GetBuffer(), mPrimaryOutput->GetImage(), VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imgCopy);
        }
        if(!!mBenchmark)
        {
            mBenchmark->CmdWriteTimestamp(cmdBuffer, frameIdx, bench::BenchmarkTimestamp::END, VkPipelineStageFlagBits::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
    }

#pragma endregion
#pragma region Destroy

    void OptiXDenoiserStage::Destroy()
    {
        DestroyResolutionDependentComponents();
        DestroyFixedComponents();
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
        if(!!mBenchmark)
        {
            mBenchmark->Destroy();
            mBenchmark = nullptr;
        }
    }

    void OptiXDenoiserStage::DestroyResolutionDependentComponents()
    {
        mScaleMotionStage.Destroy();
        for(CudaBuffer& buffer : mInputBuffers)
        {
            buffer.Destroy();
        }
        mOutputBuffer.Destroy();

        if(!!mCudaStateBuffer)
        {
            AssertCudaResult(cudaFree(reinterpret_cast<void*>(mCudaStateBuffer)));
            mCudaStateBuffer = 0;
        }
        if(!!mCudaScratchBuffer)
        {
            AssertCudaResult(cudaFree(reinterpret_cast<void*>(mCudaScratchBuffer)));
            mCudaScratchBuffer = 0;
        }
        if(!!mCudaMinRGB)
        {
            AssertCudaResult(cudaFree(reinterpret_cast<void*>(mCudaMinRGB)));
            mCudaMinRGB = 0;
        }
        if(!!mCudaIntensity)
        {
            AssertCudaResult(cudaFree(reinterpret_cast<void*>(mCudaIntensity)));
            mCudaIntensity = 0;
        }
    }

#pragma endregion
#pragma region Misc

    std::string OptiXDenoiserStage::GetUILabel()
    {
        uint32_t major = OPTIX_VERSION / 10000;
        uint32_t minor = (OPTIX_VERSION % 10000) / 100;
        uint32_t micro = OPTIX_VERSION % 100;
        return fmt::format("OptiX Denoiser v{}.{}.{}", major, minor, micro);
    }
    void OptiXDenoiserStage::DisplayImguiConfiguration() {}
    void OptiXDenoiserStage::IgnoreHistoryNextFrame()
    {
        mDenoisedFrames = 0;
    }

    void OptiXDenoiserStage::Resize(const VkExtent2D& size)
    {
        if (!mOptixDenoiser)
        {
            return;
        }
        IgnoreHistoryNextFrame();
        DestroyResolutionDependentComponents();
        CreateResolutionDependentComponents();
    }

#pragma endregion
}  // namespace foray::optix
