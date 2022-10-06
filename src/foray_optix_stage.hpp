#pragma once
#include <cuda.h>
#include <driver_types.h>
#include <memory/hsk_managedbuffer.hpp>
#include <optix_types.h>
#include <stages/hsk_denoiserstage.hpp>

namespace foray::optix {
    class OptiXDenoiserStage : public hsk::DenoiserStage
    {
      public:
        virtual void Init(const hsk::VkContext* context, const hsk::DenoiserConfig& config) override;

        virtual void BeforeDenoise(const hsk::FrameRenderInfo& renderInfo) override;
        virtual void AfterDenoise(const hsk::FrameRenderInfo& renderInfo) override;
        virtual void DispatchDenoise(uint64_t& timelineValue) override;

      protected:
        virtual void CreateFixedSizeComponents() override;
        virtual void DestroyFixedComponents() override;
        virtual void CreateResolutionDependentComponents() override;
        virtual void DestroyResolutionDependentComponents() override;

        hsk::ManagedImage* mPrimaryInput  = nullptr;
        hsk::ManagedImage* mAlbedoInput   = nullptr;
        hsk::ManagedImage* mNormalInput   = nullptr;
        hsk::ManagedImage* mPrimaryOutput = nullptr;

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
            hsk::ManagedBuffer Buffer;
#ifdef WIN32
            HANDLE Handle = {INVALID_HANDLE_VALUE};
#else
            int Handle = -1;
#endif
            void* CudaPtr = nullptr;

            void Setup(const hsk::VkContext* context);
            void Destroy();
        };

        std::array<CudaBuffer, 3> mInputBuffers;
        CudaBuffer                mOutputBuffer;

        OptixPixelFormat mPixelFormat = OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF4;
        size_t           mSizeOfPixel = 4 * sizeof(uint16_t);

        hsk::DenoiserSynchronisationSemaphore* mSemaphore;
        cudaExternalSemaphore_t                mCudaSemaphore;
    };
}  // namespace foray::optix
