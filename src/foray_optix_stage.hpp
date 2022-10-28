#pragma once
#include "foray_optix_cudabuffer.hpp"
#include <array>
#include <stages/foray_denoiserstage.hpp>
#include "foray_optix_scalemotionstage.hpp"

namespace foray::optix {

    /// @brief Complete package denoiser stage for OptiX denoise
    class OptiXDenoiserStage : public stages::ExternalDenoiserStage
    {
      public:
        virtual void Init(core::Context* context, const stages::DenoiserConfig& config) override;

        /// @brief Call this to transfer image data to synchronization buffers prior to denoising
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

        // References to Input/Output ManagedImages

        core::ManagedImage* mPrimaryInput  = nullptr;
        core::ManagedImage* mAlbedoInput   = nullptr;
        core::ManagedImage* mNormalInput   = nullptr;
        core::ManagedImage* mMotionInput   = nullptr;
        core::ManagedImage* mPrimaryOutput = nullptr;

        // Cached Denoiser Options (Guide layer selection)

        OptixDenoiserOptions mDenoiserOptions{};

        // Cuda Variables

        /// @brief Cuda Context
        CUcontext   mCudaContext{};
        /// @brief Cuda Stream
        CUstream    mCudaStream{};
        /// @brief OptiX internal state buffer
        CUdeviceptr mCudaStateBuffer{};
        /// @brief OptiX internal scratch buffer
        CUdeviceptr mCudaScratchBuffer{};
        /// @brief Single float value for intensity
        CUdeviceptr mCudaIntensity{};
        /// @brief Minimum values (4x float)
        CUdeviceptr mCudaMinRGB{};


        OptixDeviceContext mOptixDevice{};
        OptixDenoiser      mOptixDenoiser{};
        /// @brief Struct for sizes of internal optix buffers
        OptixDenoiserSizes mDenoiserSizes{};

        /// @brief Enum for indexing input buffers in mInputBuffers array
        enum EInputBufferKind
        {
            Source,
            Albedo,
            Normal,
            Motion
        };

        /// @brief Input data transfer buffers (Vulkan -> Cuda)
        std::array<CudaBuffer, 4> mInputBuffers;
        /// @brief Output data transfer buffer (Cuda -> Vulkan)
        CudaBuffer                mOutputBuffer;

        /// @brief Timeline semaphore for synchronizing Vulkan & Cuda
        util::ExternalSemaphore* mSemaphore     = nullptr;
        cudaExternalSemaphore_t  mCudaSemaphore = nullptr;

        /// @brief Counts denoised frames in a row. If zero, history is not used
        uint32_t mDenoisedFrames = 0;

        /// @brief Compute stage for scaling motion vectors into the correct values expected by OptiX
        ScaleMotionStage mScaleMotionStage;
    };
}  // namespace foray::optix
