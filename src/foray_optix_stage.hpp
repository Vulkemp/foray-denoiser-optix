#pragma once
#include <cuda.h>
#include <memory/hsk_managedbuffer.hpp>
#include <optix_types.h>
#include <stages/hsk_denoiserstage.hpp>

namespace foray::optix {
    class OptiXDenoiserStage : public hsk::DenoiserStage
    {
      public:
        virtual void Init(const VkContext* context, const hsk::DenoiserConfig& config) override;

        virtual void BeforeDenoise(const FrameRenderInfo& renderInfo) override;
        virtual void AfterDenoise(const FrameRenderInfo& renderInfo) override;
        virtual void DispatchDenoise(VkSemaphore readyToDenoise, VkSemaphore denoiseCompleted) override;

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
            HANDLE Handle = {};
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
    };
}  // namespace foray::optix
