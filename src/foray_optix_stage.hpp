#pragma once
#include <array>
#include <core/foray_managedbuffer.hpp>
#include <cuda.h>
#include <driver_types.h>
#include <optix_types.h>
#include <stages/foray_denoiserstage.hpp>

// TODO: Motion Data is useless because OptiX requires a different Motion Vector format: https://raytracing-docs.nvidia.com/optix7/guide/index.html#ai_denoiser#temporal-denoising-modes
// Ours uses normalized texture coordinates [0...1], theirs uses texel coordinates [0...width] / [0...height]
// Note: this could easily be fixed with the help of a compute shader

namespace foray::optix {
    class OptiXDenoiserStage : public stages::DenoiserStage
    {
      public:
        virtual void Init(core::Context* context, const stages::DenoiserConfig& config) override;

        virtual void BeforeDenoise(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo) override;
        virtual void AfterDenoise(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo) override;
        virtual void DispatchDenoise(uint64_t timelineValueBefore, uint64_t timelineValueAfter);

      protected:
        virtual void CreateFixedSizeComponents() override;
        virtual void DestroyFixedComponents() override;
        virtual void CreateResolutionDependentComponents() override;
        virtual void DestroyResolutionDependentComponents() override;

        core::ManagedImage* mPrimaryInput  = nullptr;
        core::ManagedImage* mAlbedoInput   = nullptr;
        core::ManagedImage* mNormalInput   = nullptr;
        core::ManagedImage* mMotionInput   = nullptr;
        core::ManagedImage* mPrimaryOutput = nullptr;

        OptixDenoiserOptions mDenoiserOptions{};

        CUcontext   mCudaContext{};
        CUstream    mCudaStream{};
        CUdeviceptr mCudaStateBuffer{};
        CUdeviceptr mCudaScratchBuffer{};
        CUdeviceptr mCudaIntensity{};
        CUdeviceptr mCudaMinRGB{};

        OptixDeviceContext mOptixDevice{};
        OptixDenoiser      mOptixDenoiser{};
        OptixDenoiserSizes mDenoiserSizes{};

        struct CudaBuffer
        {
            core::ManagedBuffer Buffer;
#ifdef WIN32
            HANDLE Handle = {INVALID_HANDLE_VALUE};
#else
            int Handle = -1;
#endif
            void* CudaPtr = nullptr;

            void Setup(core::Context* context);
            void Destroy();
        };

        enum EInputBufferKind
        {
            Source,
            Albedo,
            Normal,
            Motion
        };

        std::array<CudaBuffer, 4> mInputBuffers;
        CudaBuffer mOutputBuffer;

        OptixPixelFormat mPixelFormat = OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF4;
        OptixPixelFormat mMotionPixelFormat = OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF2;
        size_t           mSizeOfPixel = 4 * sizeof(uint16_t);

        stages::DenoiserSynchronisationSemaphore* mSemaphore     = nullptr;
        cudaExternalSemaphore_t                   mCudaSemaphore = nullptr;

        uint32_t mDenoisedFrames = 0;
    };
}  // namespace foray::optix
