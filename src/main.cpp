﻿//#define GLFW_INCLUDE_VULKAN
//#include <GLFW/glfw3.h>

//#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>

#include <array>
#include <format>
#include <fstream>
#include <iostream>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <VkBootstrap.h>
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


static constexpr uint32_t kImageWidth{ 800 };
static constexpr uint32_t kImageHeight{ 600 };

static constexpr uint32_t kWorkGroupSizeX{ 16 };
static constexpr uint32_t kWorkGroupSizeY{ 8 };


struct VulkanContextData
{
    vkb::Instance       mInstance;
    vkb::PhysicalDevice mPhysicalDevice;
    vkb::Device         mDevice;
    operator VkDevice() const { return mDevice; }

    struct Queue
    {
        VkQueue     mQueue;
        uint32_t    mIndex;
        operator VkQueue() const { return mQueue; }
    };
    Queue mQueue;
};

VulkanContextData   initVulkan();
VmaAllocator        initAllocator(VulkanContextData& context);
VkShaderModule      createShaderModule(VulkanContextData& context, const std::string& path);

int main()
{
    // Setup vulkan.
    VulkanContextData   context     = initVulkan();
    // Setup memory allocator.
    VmaAllocator        allocator   = initAllocator(context);

    // Set properties for the buffer object.
    VkDeviceSize bufferSizeBytes = kImageWidth * kImageHeight * 3 * sizeof(float);
    VkBufferCreateInfo bufferCreateInfo = {
        .sType                  = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size                   = bufferSizeBytes,
        .usage                  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode            = VK_SHARING_MODE_EXCLUSIVE
    };
    // Set memory properties for the buffer.
    VmaAllocationCreateInfo allocInfo = {
        .flags          = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage          = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags  =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  | // Specify that the buffer may be mapped. 
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | // 
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT     // Without this flag, every read of the buffer's memory requires a fetch from GPU memory!
    };

    // Create a buffer.
    VkBuffer buffer;
    VmaAllocation bufferAllocation;
    vmaCreateBuffer(allocator, &bufferCreateInfo, &allocInfo, &buffer, &bufferAllocation, nullptr);

    // Create shader module.
    VkShaderModule computeShaderModule = createShaderModule(context, "shaders/rayTrace.spv");

    // Assign shader module to compute shader stage.
    VkPipelineShaderStageCreateInfo computeShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = computeShaderModule,
        .pName = "main"
    };

    // Configure descriptor sets.
    
    // Specify the type (SSBO) and binding location (0) of the single entry in the descriptor set we will use.
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding0{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
    };
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &descriptorSetLayoutBinding0
    };
    VkDescriptorSetLayout descriptorSetLayout;
    auto result = vkCreateDescriptorSetLayout(context, &descriptorSetLayoutInfo, nullptr, &descriptorSetLayout);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create descriptor set layout!: {}\n", string_VkResult(result));
    }

    // Create descriptor pool.
    // For our app, the only type of resource we be using is SSBO, and will only need at most 1 combined over all sets. 
    VkDescriptorPoolSize SSBOPoolSize{
        .type               = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount    = 1
    };
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{
        .sType          = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets        = 1, //Only need one descriptor set to be allocated.
        .poolSizeCount  = 1,
        .pPoolSizes     = &SSBOPoolSize
    };
    VkDescriptorPool descriptorPool;
    result = vkCreateDescriptorPool(context, &descriptorPoolCreateInfo, nullptr, &descriptorPool);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create descriptor set pool!: {}\n", string_VkResult(result));
    }

    // Allocate descriptor set from the pool.
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout
    };
    VkDescriptorSet descriptorSet;
    result = vkAllocateDescriptorSets(context, &descriptorSetAllocateInfo,&descriptorSet);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to allocate descriptor set!: {}\n", string_VkResult(result));
    }

    // Update descriptor with the buffer we created.
    VkDescriptorBufferInfo descriptorBufferInfo{
        .buffer = buffer,    // The VkBuffer object
        .offset = 0,
        .range  = bufferSizeBytes  // The length of memory to bind; offset is 0.
    };
    // Fill the descriptor sets with resources.
    VkWriteDescriptorSet descriptorWrite{
        .sType              = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet             = descriptorSet,
        .dstBinding         = 0,
        .dstArrayElement    = 0,
        .descriptorCount    = 1,
        .descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &descriptorBufferInfo
    };
    vkUpdateDescriptorSets(
        context,
        1, &descriptorWrite,  // An array of VkWriteDescriptorSet objects
        0, nullptr);          // An array of VkCopyDescriptorSet objects (unused)


    // Create the compute pipeline for the app.
    // A compute pipeline is essentially just a compute shader.

    // Layout for the pipeline.
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &descriptorSetLayout,
        .pushConstantRangeCount = 0
    };
    VkPipelineLayout computePipelineLayout;
    result = vkCreatePipelineLayout(context, &pipelineLayoutCreateInfo, VK_NULL_HANDLE, &computePipelineLayout);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create compute pipeline layout!: {}\n", string_VkResult(result));
    }
    // Create the pipeline.
    VkComputePipelineCreateInfo pipelineCreateInfo{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = computeShaderStageCreateInfo,
        .layout = computePipelineLayout
    }; 
    VkPipeline computePipeline;
    vkCreateComputePipelines(context,                 // Device
        VK_NULL_HANDLE,          // Pipeline cache (uses default)
        1, &pipelineCreateInfo,  // Compute pipeline create info
        VK_NULL_HANDLE,          // Allocator (uses default)
        &computePipeline
    );      // Output



    // Create command pool for the compute queue.
    VkCommandPoolCreateInfo commanddPoolCreateInfo = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex   = context.mQueue.mIndex,
    };
    VkCommandPool commandPool;
    result = vkCreateCommandPool(context, &commanddPoolCreateInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create Command pool: {}\n", string_VkResult(result));
    }
    std::cout << "Command pool created successfully!\n";

    //Allocate a command buffer from the pool.
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = commandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VkCommandBuffer commandBuffer;
    result = vkAllocateCommandBuffers(context, &commandBufferAllocateInfo,&commandBuffer);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create Command buffer: {}\n", string_VkResult(result));
    }
    std::cout << "Command buffer created successfully!\n";


    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT // After submitting it we will re-record it before submitting again.
    };
    result = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to begin recording Command buffer: {}\n", string_VkResult(result));
    }
    std::cout << "Recording has begun!\n";

    // Bind the compute pipeline and dispatch the compute shader.
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    // Bind the descriptor set
    vkCmdBindDescriptorSets(commandBuffer, 
        VK_PIPELINE_BIND_POINT_COMPUTE,
        computePipelineLayout,
        0, 1,               //Index of starting set to bind to
        &descriptorSet,     //Pointer to array of sets to bind.
        0, nullptr);
    vkCmdDispatch(commandBuffer, (uint32_t(kImageWidth) + kWorkGroupSizeX - 1) / kWorkGroupSizeX,
        (uint32_t(kImageHeight) + kWorkGroupSizeY- 1) / kWorkGroupSizeY, 1);


    // Insert a pipeline barrier that ensures GPU memory writes are available for the CPU to read.
    VkMemoryBarrier memoryBarrier = {
        .sType          = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask  = VK_ACCESS_SHADER_WRITE_BIT,  // Make shader writes
        .dstAccessMask  = VK_ACCESS_HOST_READ_BIT       // Readable by the CPU
    };
    vkCmdPipelineBarrier(commandBuffer,                                // The command buffer
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,           // From the transfer stage
        VK_PIPELINE_STAGE_HOST_BIT,               // To the CPU
        0,                                        // No special flags
        1, &memoryBarrier,                        // Pass the single global memory barrier.
        0, nullptr, 0, nullptr);                  // No image/buffer memory barriers.

    // End recording of command buffer
    result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to end recording Command buffer: {}\n", string_VkResult(result));
    }
    std::cout << "Recording has ended!\n";

    // Submit command buffer to queue.
    VkSubmitInfo submitInfo = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &commandBuffer
    };
    result = vkQueueSubmit(context.mQueue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to submit Command buffer to queue: {}\n", string_VkResult(result));
    }
    std::cout << "Command buffer has been submitted to compute queue!\n";

    // Synchronise:  Block until queue has no more work to do.
    // Note that it is not enough alone. This is why we inserted a pipeline barrier.
    result = vkQueueWaitIdle(context.mQueue);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to finish processing compute queue: {}\n", string_VkResult(result));
    }
    std::cout << "Compute queue has finished processing command buffer!\n";

    // Map data from GPU to CPU to read it.
    void* pData;
    vmaMapMemory(allocator, bufferAllocation, &pData);
    stbi_write_hdr("out.hdr", kImageWidth, kImageHeight, 3, reinterpret_cast<float*>(pData));
    vmaUnmapMemory(allocator, bufferAllocation);



    vkDestroyPipeline(context, computePipeline, nullptr);
    vkDestroyShaderModule(context, computeShaderModule, nullptr);
    vkDestroyPipelineLayout(context, computePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(context, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(context, descriptorPool, nullptr);
    vkFreeCommandBuffers(context, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(context, commandPool, nullptr);
    vmaDestroyBuffer(allocator, buffer, bufferAllocation);
    vmaDestroyAllocator(allocator);
    vkb::destroy_device(context.mDevice);
    vkb::destroy_instance(context.mInstance);
}



VulkanContextData initVulkan()
{
    // Create instance
    vkb::InstanceBuilder builder;
    const auto inst_ret = builder.set_app_name("Hello Vulkan")
        .request_validation_layers(true)
        .enable_validation_layers()
        .require_api_version(1, 2)
        .use_default_debug_messenger()
        .build();
    if (!inst_ret) {
        std::cerr << std::format("Failed to create Vulkan instance: {}\n", inst_ret.error().message());
    }
    vkb::Instance vkb_instance = inst_ret.value();
    std::cout << std::format("Vulkan instance created successfully!\n");


    // Select a physical device that supports the required extensions
    // Note that we don't need a surface for this project. 
    vkb::PhysicalDeviceSelector physDeviceSelector{ vkb_instance };
    const auto physDevice_ret = physDeviceSelector
        .set_minimum_version(1, 2)
        .defer_surface_initialization()
        .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
        .select();
    if (!physDevice_ret) {
        std::cerr << std::format("Failed to create Vulkan physical device: {}\n", physDevice_ret.error().message());
    }
    vkb::PhysicalDevice vkb_physical_device = physDevice_ret.value();
    std::cout << "Vulkan physical device selected successfully!\n";

    // Create logical device.
    vkb::DeviceBuilder device_builder(vkb_physical_device);
    const auto dev_ret = device_builder
        .build();
    if (!dev_ret) {
        std::cerr << std::format("Failed to create Vulkan logical device: {}\n", dev_ret.error().message());
    }
    vkb::Device vkb_device = dev_ret.value();
    std::cout << "Vulkan logical device created successfully!\n";

    // Search device for a compute queue.
    const auto queue_ret = vkb_device.get_queue(vkb::QueueType::compute);
    if (!queue_ret) {
        std::cerr << std::format("Failed to find a compute queue: {}\n", queue_ret.error().message());
    }
    VkQueue compute_queue = queue_ret.value();
    const auto compute_queue_index = vkb_device.get_queue_index(vkb::QueueType::compute).value();
    std::cout << std::format("Compute queue found at index {}!\n", compute_queue_index);

    return { vkb_instance,vkb_physical_device,vkb_device, {compute_queue, compute_queue_index } };
}



VmaAllocator initAllocator(VulkanContextData& context)
{
    // Create memory allocator.
    VmaAllocatorCreateInfo allocatorInfo = {
        .physicalDevice = context.mPhysicalDevice,
        .device = context.mDevice,
        .instance = context.mInstance
    };
    VmaAllocator allocator;
    auto result = vmaCreateAllocator(&allocatorInfo, &allocator);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create VmaAllocator: {}\n", string_VkResult(result));
    }
    std::cout << "VMA Allocator created successfully!\n";
    return allocator;
}


// Loads binary data from a file
std::vector<char> readBinaryFile(const std::string& filename) 
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
}

VkShaderModule createShaderModule(VulkanContextData& context, const std::string& path)
{
    // Create shader module.
    const auto shaderByteCode = readBinaryFile(path);
    VkShaderModuleCreateInfo shaderModuleCreateInfo{
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .flags = {},
    .codeSize = shaderByteCode.size(),
    .pCode = reinterpret_cast<const uint32_t*>(shaderByteCode.data())
    };
    VkShaderModule shaderModule;
    auto result = vkCreateShaderModule(context, &shaderModuleCreateInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create shader module: {}\n", string_VkResult(result));
    }
    std::cout << "Shader module created successfully!\n";
    return shaderModule;
}
