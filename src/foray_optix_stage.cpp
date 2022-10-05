#include "foray_optix_stage.hpp"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include "foray_optix_helpers.hpp"

namespace foray::optix
{
    void OptiXDenoiserStage::Init(hsk::ManagedImage *noisy, hsk::ManagedImage *baseColor, hsk::ManagedImage *normal, bool useTemporal)
    {
        AssertCudaResult(cuInit(0)); // Initialize CUDA driver API.

        CUdevice device = 0;
        AssertCudaResult(cuCtxCreate(&mCudaContext, CU_CTX_SCHED_SPIN, device));

        // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
        AssertCudaResult(cuStreamCreate(&mCudaStream, CU_STREAM_DEFAULT));

        AssertOptiXResult(optixInit());
        AssertOptiXResult(optixDeviceContextCreate(mCudaContext, nullptr, &mOptixDevice));
    }

    void OptiXDenoiserStage::RecordFrame(hsk::FrameRenderInfo &renderInfo) {}
    void OptiXDenoiserStage::CreateFixedSizeComponents() {}
    void OptiXDenoiserStage::DestroyFixedComponents() {}
    void OptiXDenoiserStage::CreateResolutionDependentComponents() {}
    void OptiXDenoiserStage::DestroyResolutionDependentComponents() {}

} // namespace foray::optix
