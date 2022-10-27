#pragma once
#include <stages/foray_computestage.hpp>

namespace foray::optix {

    class ScaleMotionStage : public stages::ComputeStage
    {
      public:
        void Init(core::Context* context, core::ManagedImage* input, core::ManagedBuffer* output);

      protected:
        virtual void ApiInitShader();
        virtual void ApiCreateDescriptorSetLayout();
        virtual void ApiCreatePipelineLayout();
        
        virtual void ApiBeforeFrame(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo);
        virtual void ApiBeforeDispatch(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo, glm::uvec3& groupSize);

        core::ManagedImage*  mInput  = nullptr;
        core::ManagedBuffer* mOutput = nullptr;
    };
}  // namespace foray::optix
