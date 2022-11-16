#include "foray_optix_scalemotionstage.hpp"
#include <core/foray_managedbuffer.hpp>
#include <core/foray_managedimage.hpp>


const uint32_t SCALEMOTION_SPIRV[] =
#include "foray_optix_scalemotion.comp.spv.h"
    ;

namespace foray::optix {
    void ScaleMotionStage::Init(core::Context* context, core::ManagedImage* input, core::ManagedBuffer* output)
    {
        mInput  = input;
        mOutput = output;
        stages::ComputeStage::Init(context);
    }
    void ScaleMotionStage::ApiInitShader()
    {
        mShader.LoadFromBinary(mContext, SCALEMOTION_SPIRV, sizeof(SCALEMOTION_SPIRV));
    }
    void ScaleMotionStage::ApiCreateDescriptorSet()
    {
        {  // Input Descriptor Info
            mDescriptorSet.SetDescriptorAt(0, mInput, VkImageLayout::VK_IMAGE_LAYOUT_GENERAL, nullptr, VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                           VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
        }

        {  // Output Descriptor Info
            mDescriptorSet.SetDescriptorAt(1, mOutput, VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
        }

        mDescriptorSet.Create(mContext, "OptiX ScaleMotionStage Descriptor Set");
    }
    void ScaleMotionStage::ApiCreatePipelineLayout()
    {
        mPipelineLayout.AddDescriptorSetLayout(mDescriptorSet.GetDescriptorSetLayout());
        mPipelineLayout.AddPushConstantRange<VkExtent2D>(VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT);
        mPipelineLayout.Build(mContext);
    }
    void ScaleMotionStage::ApiBeforeFrame(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo)
    {
        core::ImageLayoutCache::Barrier2 imageBarrier{.SrcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                      .SrcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                                      .DstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                      .DstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                                      .NewLayout     = VK_IMAGE_LAYOUT_GENERAL};
        VkImageMemoryBarrier2            vkImageBarrier = renderInfo.GetImageLayoutCache().MakeBarrier(mInput, imageBarrier);

        VkBufferMemoryBarrier2 bufferBarrier{.sType         = VkStructureType::VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                             .srcStageMask  = VK_PIPELINE_STAGE_2_NONE,
                                             .srcAccessMask = VK_ACCESS_2_NONE,
                                             .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                             .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                             .buffer        = mOutput->GetBuffer(),
                                             .offset        = 0,
                                             .size          = VK_WHOLE_SIZE};

        VkDependencyInfo depInfo{
            .sType                    = VkStructureType::VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = 1U,
            .pBufferMemoryBarriers    = &bufferBarrier,
            .imageMemoryBarrierCount  = 1U,
            .pImageMemoryBarriers     = &vkImageBarrier,
        };

        mContext->VkbDispatchTable->cmdPipelineBarrier2(cmdBuffer, &depInfo);
    }
    void ScaleMotionStage::ApiBeforeDispatch(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo, glm::uvec3& groupSize)
    {
        const glm::uvec2 localSize = glm::uvec2(16, 16);

        VkExtent2D screenSize = mContext->GetSwapchainSize();
        groupSize.x           = (screenSize.width + localSize.x - 1) / localSize.x;
        groupSize.y           = (screenSize.height + localSize.y - 1) / localSize.y;
        groupSize.z           = 1;

        mContext->VkbDispatchTable->cmdPushConstants(cmdBuffer, mPipelineLayout, VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkExtent2D), &screenSize);
    }

}  // namespace foray::optix
