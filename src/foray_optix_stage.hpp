#pragma once
#include <stages/hsk_denoiserstage.hpp>
#include <optix_types.h>
#include <cuda.h>

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
        OptixDeviceContext mOptixDevice{};
    };
} // namespace foray::optix
