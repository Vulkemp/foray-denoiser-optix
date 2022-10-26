#pragma once
#include "foray_optix_cudabuffer.hpp"
#include <array>
#include <stages/foray_denoiserstage.hpp>
#include "foray_optix_scalemotionstage.hpp"

// TODO: Motion Data is useless because OptiX requires a different Motion Vector format: https://raytracing-docs.nvidia.com/optix7/guide/index.html#ai_denoiser#temporal-denoising-modes
// Ours uses normalized texture coordinates [0...1], theirs uses texel coordinates [0...width] / [0...height]
// Note: this could easily be fixed with the help of a compute shader

namespace foray::optix {
    class OptiXDenoiserStage : public stages::ExternalDenoiserStage
    {
      public:
        virtual void Init(core::Context* context, const stages::DenoiserConfig& config) override;

        virtual void BeforeDenoise(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo) override;
        virtual void AfterDenoise(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo) override;
        virtual void DispatchDenoise(uint64_t timelineValueBefore, uint64_t timelineValueAfter) override;

        virtual void OnResized(const VkExtent2D& size) override;

        virtual std::string GetUILabel() override;
        virtual void        DisplayImguiConfiguration() override;
        virtual void        IgnoreHistoryNextFrame() override;

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

        enum EInputBufferKind
        {
            Source,
            Albedo,
            Normal,
            Motion
        };

        std::array<CudaBuffer, 4> mInputBuffers;
        CudaBuffer                mOutputBuffer;

        util::ExternalSemaphore* mSemaphore     = nullptr;
        cudaExternalSemaphore_t  mCudaSemaphore = nullptr;

        uint32_t mDenoisedFrames = 0;

        ScaleMotionStage mScaleMotionStage;
    };
}  // namespace foray::optix
