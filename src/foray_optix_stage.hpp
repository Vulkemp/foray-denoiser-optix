#pragma once
#include <array>
#include <core/foray_managedbuffer.hpp>
#include <cuda.h>
#include <driver_types.h>
#include <optix_types.h>
#include <stages/foray_denoiserstage.hpp>

namespace foray::optix {
    class OptiXDenoiserStage : public stages::DenoiserStage
    {
      public:
        virtual void Init(const core::VkContext* context, const stages::DenoiserConfig& config) override;

        virtual void BeforeDenoise(const base::FrameRenderInfo& renderInfo) override;
        virtual void AfterDenoise(const base::FrameRenderInfo& renderInfo) override;
        virtual void DispatchDenoise(uint64_t& timelineValue) override;

      protected:
        virtual void CreateFixedSizeComponents() override;
        virtual void DestroyFixedComponents() override;
        virtual void CreateResolutionDependentComponents() override;
        virtual void DestroyResolutionDependentComponents() override;

        core::ManagedImage* mPrimaryInput  = nullptr;
        core::ManagedImage* mAlbedoInput   = nullptr;
        core::ManagedImage* mNormalInput   = nullptr;
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

            void Setup(const core::VkContext* context);
            void Destroy();
        };

        std::array<CudaBuffer, 3> mInputBuffers;
        CudaBuffer                mOutputBuffer;

        OptixPixelFormat mPixelFormat = OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF4;
        size_t           mSizeOfPixel = 4 * sizeof(uint16_t);

        stages::DenoiserSynchronisationSemaphore* mSemaphore;
        cudaExternalSemaphore_t                   mCudaSemaphore;
    };
}  // namespace foray::optix
