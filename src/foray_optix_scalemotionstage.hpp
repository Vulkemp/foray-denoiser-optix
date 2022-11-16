#pragma once
#include <stages/foray_computestage.hpp>

namespace foray::optix {

    class ScaleMotionStage : public stages::ComputeStage
    {
      public:
        void Init(core::Context* context, core::ManagedImage* input, core::ManagedBuffer* output);

      protected:
        virtual void ApiInitShader() override;
        virtual void ApiCreateDescriptorSet() override;
        virtual void ApiCreatePipelineLayout() override;
        
        virtual void ApiBeforeFrame(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo) override;
        virtual void ApiBeforeDispatch(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo, glm::uvec3& groupSize) override;

        core::ManagedImage*  mInput  = nullptr;
        core::ManagedBuffer* mOutput = nullptr;
    };
}  // namespace foray::optix
