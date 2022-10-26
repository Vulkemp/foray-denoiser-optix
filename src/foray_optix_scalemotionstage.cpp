#include "foray_optix_scalemotionstage.hpp"
#include <core/foray_managedbuffer.hpp>
#include <core/foray_managedimage.hpp>


const uint32_t SCALEMOTION_SPIRV[] =
#include "scalemotion.spv.h"
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
        std::vector<char> binary(sizeof(SCALEMOTION_SPIRV));
        memcpy(binary.data(), SCALEMOTION_SPIRV, sizeof(SCALEMOTION_SPIRV));
        mShader.LoadFromBinary(mContext, binary);
    }
    void ScaleMotionStage::ApiCreateDescriptorSetLayout()
    {
        {  // Update Info Vectors
            mInputImageInfo   = {VkDescriptorImageInfo{.sampler = nullptr, .imageView = mInput->GetImageView(), .imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_GENERAL}};
            mOutputBufferInfo = {mOutput->GetVkDescriptorBufferInfo()};
        }

        {  // Input Descriptor Info
            std::shared_ptr<core::DescriptorSetHelper::DescriptorInfo> inputDescriptorInfo = std::make_shared<core::DescriptorSetHelper::DescriptorInfo>();
            inputDescriptorInfo->Init(VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT, &mInputImageInfo);
            mDescriptorSet.SetDescriptorInfoAt(0, inputDescriptorInfo);
        }

        {  // Output Descriptor Info
            std::shared_ptr<core::DescriptorSetHelper::DescriptorInfo> outputDescriptorInfo = std::make_shared<core::DescriptorSetHelper::DescriptorInfo>();
            outputDescriptorInfo->Init(VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT, &mOutputBufferInfo);
            mDescriptorSet.SetDescriptorInfoAt(1, outputDescriptorInfo);
        }

        mDescriptorSet.Create(mContext, -1, "ScaleMotionStage Descriptor Set");
    }
    void ScaleMotionStage::ApiCreatePipelineLayout()
    {
        mPipelineLayout.AddDescriptorSetLayout(mDescriptorSet.GetDescriptorSetLayout());
        mPipelineLayout.AddPushConstantRange<VkExtent2D>();
        mPipelineLayout.Create(mContext);
    }
    void ScaleMotionStage::ApiBeforeFrame(VkCommandBuffer cmdBuffer, base::FrameRenderInfo& renderInfo)
    {
        core::ImageLayoutCache::Barrier2 imageBarrier{.SrcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                      .SrcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                                      .DstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                      .DstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                                      .NewLayout     = VK_IMAGE_LAYOUT_GENERAL};
        VkImageMemoryBarrier2            vkImageBarrier = renderInfo.GetImageLayoutCache().Set(mInput, imageBarrier);

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
