#pragma once
#include <stages/hsk_denoiserstage.hpp>
#include <optix_types.h>
#include <cuda.h>
#include <memory/hsk_managedbuffer.hpp>

namespace foray::optix
{
    class OptiXDenoiserStage : public hsk::DenoiserStage
    {
    public:
        void Init(hsk::ManagedImage *noisy, hsk::ManagedImage *baseColor, hsk::ManagedImage *normal, bool useTemporal = true);

        virtual void RecordFrame(hsk::FrameRenderInfo &renderInfo) override;

    protected:
        virtual void CreateFixedSizeComponents() override;
        virtual void DestroyFixedComponents() override;
        virtual void CreateResolutionDependentComponents() override;
        virtual void DestroyResolutionDependentComponents() override;

        hsk::ManagedImage mTemporal;
        OptixDenoiserOptions mDenoiserOptions{};

        CUcontext mCudaContext{};
        CUstream mCudaStream{};
        CUdeviceptr mCudaStateBuffer{};
        CUdeviceptr mCudaScratchBuffer{};
        CUdeviceptr mCudaIntensity{};
        CUdeviceptr mCudaMinRGB{};

        OptixDeviceContext mOptixDevice{};
        OptixDenoiser mOptixDenoiser{};
        OptixDenoiserSizes mDenoiserSizes{};

        struct CudaBuffer
        {
            hsk::ManagedBuffer Buffer;
#ifdef WIN32
            HANDLE Handle = {};
#else
            int Handle = -1;
#endif
            void *CudaPtr = nullptr;

            void Setup(const hsk::VkContext *context);
            void Destroy();
        };

        std::array<CudaBuffer, 3> mInputBuffers;
        CudaBuffer mOutputBuffer;

        size_t mSizeOfPixel = 4 * sizeof(uint16_t);
    };
} // namespace foray::optix
