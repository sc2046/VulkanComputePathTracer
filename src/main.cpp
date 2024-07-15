//#define GLFW_INCLUDE_VULKAN
//#include <GLFW/glfw3.h>

//#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>

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

// Loads binary data from a file
std::vector<char> readFile(const std::string& filename) {
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




int main()
{
    // Create Vulkan instance using vk-bootstrap
    vkb::InstanceBuilder builder;

    auto inst_ret = builder.set_app_name("Hello Vulkan")
        .request_validation_layers(true)
        .enable_validation_layers()
        .require_api_version(1, 2)
        .use_default_debug_messenger()
        .build();

    if (!inst_ret) {
        std::cerr << std::format("Failed to create Vulkan instance: {}\n", inst_ret.error().message());
        return EXIT_FAILURE;
    }

    vkb::Instance vkb_instance = inst_ret.value();
    VkInstance instance = vkb_instance.instance;
    VkDebugUtilsMessengerEXT debug_messenger = vkb_instance.debug_messenger;

    std::cout << std::format("Vulkan instance created successfully using vk-bootstrap!\n");


    // Select a physical device that supports the required extensions
    // Note that we don't need a surface for this project. 
    vkb::PhysicalDeviceSelector physDeviceSelector{ vkb_instance };
    auto physDevice_ret = physDeviceSelector
        .set_minimum_version(1, 2)
        .defer_surface_initialization() 
        .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
        .select();

    if (!physDevice_ret) {
        std::cerr << std::format("Failed to create Vulkan physical device: {}\n", physDevice_ret.error().message());
        return EXIT_FAILURE;
    }

    vkb::PhysicalDevice vkb_physical_device = physDevice_ret.value();
    VkPhysicalDevice physicalDevice = vkb_physical_device.physical_device;

    std::cout << "Vulkan physical device selected successfully!\n";

    // Create logical device.
    vkb::DeviceBuilder device_builder(vkb_physical_device);
    auto dev_ret = device_builder
        .build();
    if (!dev_ret) {
        std::cerr << std::format("Failed to create Vulkan logical device: {}\n", dev_ret.error().message());
    }
    vkb::Device vkb_device = dev_ret.value();
    VkDevice device = vkb_device.device;
    std::cout << "Vulkan logical device created successfully!\n";

    // Create memory allocator.
    VmaAllocatorCreateInfo allocatorInfo = {
        .physicalDevice = physicalDevice,
        .device         = device,
        .instance       = instance
    };
    VmaAllocator allocator;
    auto result = vmaCreateAllocator(&allocatorInfo, &allocator);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create VmaAllocator: {}\n", string_VkResult(result));
    }
    std::cout << "VMA Allocator created successfully!\n";

    // Get queue from device
    auto queue_ret = vkb_device.get_queue(vkb::QueueType::compute);
    if (!queue_ret) {
        std::cerr << std::format("Failed to find a compute queue: {}\n", queue_ret.error().message());
    }
    VkQueue compute_queue = queue_ret.value();
    std::cout << "Compute queue found!\n";

    auto compute_queue_index = vkb_device.get_queue_index(vkb::QueueType::compute).value();

    // Set properties for the buffer.
    VkDeviceSize bufferSizeBytes = kImageWidth * kImageHeight * 3 * sizeof(float);
    VkBufferCreateInfo bufferCreateInfo = {
        .sType                  = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size                   = bufferSizeBytes,
        .usage                  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode            = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount  = 1,
        .pQueueFamilyIndices    = &compute_queue_index
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
    const auto shaderByteCode = readFile("shaders/compute.spv");
    VkShaderModuleCreateInfo shaderModuleCreateInfo{
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .flags = {},
    .codeSize = shaderByteCode.size(),
    .pCode = reinterpret_cast<const uint32_t*>(shaderByteCode.data())
    };
    VkShaderModule computeShaderModule;
    if (vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &computeShaderModule) != VK_SUCCESS) {
        std::cerr << std::format("Failed to create shader module: {}\n", string_VkResult(result));
    }
    std::cout << "Shader module created successfully!\n";

    // Assign shader module to compute shader stage.
    VkPipelineShaderStageCreateInfo computeShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = computeShaderModule,
        .pName = "main"
    };

    // Create the compute pipeline for the app.
    // A compute pipeline is essentially just a compute shader.

    // Layout for the pipeline.
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0,
        .pushConstantRangeCount = 0
    };
    VkPipelineLayout computePipelineLayout;
    result = vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, VK_NULL_HANDLE, &computePipelineLayout);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create compute pipeline layout!: {}\n", string_VkResult(result));
    }
    // Create the pipeline.
    VkComputePipelineCreateInfo pipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = computeShaderStageCreateInfo,
        .layout = computePipelineLayout
    }; 
    VkPipeline computePipeline;
    vkCreateComputePipelines(device,                 // Device
        VK_NULL_HANDLE,          // Pipeline cache (uses default)
        1, &pipelineCreateInfo,  // Compute pipeline create info
        VK_NULL_HANDLE,          // Allocator (uses default)
        &computePipeline);      // Output

    // Create command pool for the compute queue.
    VkCommandPoolCreateInfo commanddPoolCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = compute_queue_index,
    };
    VkCommandPool commandPool;
    result = vkCreateCommandPool(device, &commanddPoolCreateInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create Command pool: {}\n", string_VkResult(result));
    }
    std::cout << "Command pool created successfully!\n";

    //Allocate a command buffer from the pool.
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VkCommandBuffer commandBuffer;
    result = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo,&commandBuffer);
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
    vkCmdDispatch(commandBuffer, 1, 1, 1);


    // Insert a pipeline barrier that ensures GPU memory writes are available for the CPU to read.
    VkMemoryBarrier memoryBarrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,  // Make shader writes
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT       // Readable by the CPU
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
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer
    };
    result = vkQueueSubmit(compute_queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to submit Command buffer to queue: {}\n", string_VkResult(result));
    }
    std::cout << "Command buffer has been submitted to compute queue!\n";

    // Synchronise:  Block until queue has no more work to do.
    // Note that it is not enough alone. This is why we inserted a pipeline barrier.
    result = vkQueueWaitIdle(compute_queue);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to finish processing compute queue: {}\n", string_VkResult(result));
    }
    std::cout << "Compute queue has finished processing command buffer!\n";

    // Map data from GPU to CPU to read it.
    void* pData;
    vmaMapMemory(allocator, bufferAllocation, &pData);
    stbi_write_hdr("out.hdr", kImageWidth, kImageHeight, 3, reinterpret_cast<float*>(pData));
    vmaUnmapMemory(allocator, bufferAllocation);



    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyShaderModule(device, computeShaderModule, nullptr);
    vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);  
    vkFreeCommandBuffers(vkb_device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(vkb_device, commandPool, nullptr);
    vmaDestroyBuffer(allocator, buffer, bufferAllocation);
    vmaDestroyAllocator(allocator);
    vkb::destroy_device(vkb_device);
    vkb::destroy_instance(vkb_instance);
}
