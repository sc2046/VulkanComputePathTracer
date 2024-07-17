//#define GLFW_INCLUDE_VULKAN
//#include <GLFW/glfw3.h>

//#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>

#include <array>
#include <format>
#include <fstream>
#include <iostream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

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

struct MeshData
{
    std::vector<float>		mVertices;
    std::vector<uint32_t>	mIndices;
};

VulkanContextData   initVulkan();
VmaAllocator        initAllocator(VulkanContextData& context);
VkShaderModule      createShaderModule(VkDevice device, const std::string& path);
MeshData            LoadMeshFromObj(const std::string& path);
VkCommandBuffer     AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool);
void                EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer);
VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer);

int main()
{
    // Setup vulkan.
    VulkanContextData   context     = initVulkan();
    // Setup memory allocator.
    VmaAllocator        allocator   = initAllocator(context);

    /*
    * ==============================================================================================
    * Create a buffer that will eventually be used as a SSBO for the compute shader.
    * ==============================================================================================
    */
    
    VkBuffer buffer;
    VmaAllocation bufferAllocation;
    const VkDeviceSize ssboSizeBytes = kImageWidth * kImageHeight * 3 * sizeof(float);
    {
        VkBufferCreateInfo bufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = ssboSizeBytes,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };
        // Set memory properties for the buffer.
        // The ssbo will be accessed by the CPU and GPU, so we want it to be host-visible.
        VmaAllocationCreateInfo allocInfo{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO, // Let the allocator decide which memory type to use.
            .requiredFlags =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | // Specify that the buffer may be mapped. 
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | // 
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT     // Without this flag, every read of the buffer's memory requires a fetch from GPU memory!
        };
        // Construct the buffer.
        vmaCreateBuffer(allocator, &bufferCreateInfo, &allocInfo, &buffer, &bufferAllocation, nullptr);
    }

    /*
    * ==============================================================================================
    * Create a command pool.
    * ==============================================================================================
    */
    VkCommandPool commandPool;
    {
        // Create command pool for the compute queue.
        VkCommandPoolCreateInfo commanddPoolCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = context.mQueue.mIndex,
        };
        auto result = vkCreateCommandPool(context, &commanddPoolCreateInfo, nullptr, &commandPool);
        if (result != VK_SUCCESS) {
            std::cerr << std::format("Failed to create Command pool: {}\n", string_VkResult(result));
        }
    }

    
    /*
    * ==============================================================================================
    * Load the obj file.
    * Create another two buffers that will store the vertices and indices of the obj model.
    * ==============================================================================================
    */

    const auto [vertices, indices] {LoadMeshFromObj("assets/cornell.obj")};
    VkBuffer vertexBuffer;
    VmaAllocation vertexBufferAllocation;
    const VkDeviceSize vboSizeBytes = vertices.size() * sizeof(float);

    VkBuffer indexBuffer;
    VmaAllocation indexBufferAllocation;
    const VkDeviceSize iboSizeBytes = indices.size() * sizeof(uint32_t);

    /*
    * ==============================================================================================
    * Upload the vertices and indices to GPU in device-local buffers with the use of a staging buffer.
    * ==============================================================================================
    */
    {
        // we need to create a command buffer to copy data from the staging buffer to the device-local buffers.
        VkCommandBuffer uploadCmdBuffer = AllocateAndBeginOneTimeCommandBuffer(context, commandPool);

        // Construct the device-local vertex and index buffers.
        {
            const VkBufferUsageFlags deviceLocalUsage =
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
            VkBufferCreateInfo deviceLocalBufferCreateInfo = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = vboSizeBytes,
                .usage = deviceLocalUsage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE
            };
            // Set memory properties for the buffers.
            VmaAllocationCreateInfo deviceLocalBufferAllocInfo{
                .usage = VMA_MEMORY_USAGE_AUTO,
                .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            };
            vmaCreateBuffer(allocator, &deviceLocalBufferCreateInfo, &deviceLocalBufferAllocInfo, &vertexBuffer, &vertexBufferAllocation, nullptr);
            deviceLocalBufferCreateInfo.size = iboSizeBytes;
            vmaCreateBuffer(allocator, &deviceLocalBufferCreateInfo, &deviceLocalBufferAllocInfo, &indexBuffer, &indexBufferAllocation, nullptr);
        }

        // Create a staging buffer.
        VkBuffer stagingBuffer;
        VmaAllocation stagingBufferAllocation;
        {
            VkBufferCreateInfo stagingBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = std::max(iboSizeBytes, vboSizeBytes), // Large enough to store data for both ibo/vbo.
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE
            };
            // Set memory properties for the buffer.
            VmaAllocationCreateInfo stagingBufferAllocCreateInfo{
                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
                .requiredFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            };
            vmaCreateBuffer(allocator, &stagingBufferCreateInfo, &stagingBufferAllocCreateInfo, &stagingBuffer, &stagingBufferAllocation, nullptr);
        }
        void* stageMapping;
        vmaMapMemory(allocator, stagingBufferAllocation, &stageMapping);

        // Copy vertex data to staging buffer.
        // Transfer it from staging buffer to device-local buffers.
        memcpy(stageMapping, vertices.data(), vertices.size());
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0; // Optional
        copyRegion.dstOffset = 0; // Optional
        copyRegion.size = vertices.size();
        vkCmdCopyBuffer(uploadCmdBuffer, stagingBuffer, vertexBuffer, 1, &copyRegion);

        //Repeat for index data.
        memcpy(stageMapping, indices.data(), indices.size()); // copy to staging buffer.
        copyRegion.size = indices.size();
        vkCmdCopyBuffer(uploadCmdBuffer, stagingBuffer, indexBuffer, 1, &copyRegion);


        // Finish recording.
        EndSubmitWaitAndFreeCommandBuffer(context, context.mQueue, commandPool, uploadCmdBuffer);
        
        // Destroy resources for the staging buffer as it is no longer needed.
        vmaUnmapMemory(allocator, stagingBufferAllocation);
        vmaDestroyBuffer(allocator, stagingBuffer, stagingBufferAllocation);
    }

    /*
    * ==============================================================================================
    * Load compute shader data from file.
    * ==============================================================================================
    */
    VkShaderModule computeShaderModule = createShaderModule(context, "shaders/rayTrace.spv");
    // Assign module to the compute stage.
    VkPipelineShaderStageCreateInfo computeShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = computeShaderModule,
        .pName = "main"
    };

    /*
    * ==============================================================================================
    * Configure descriptor set layouts. 
    * For now (ch. 1-7) our pipeline only uses a single resource, the SSBO that image data is written to.
    * ==============================================================================================
    */
    VkDescriptorSetLayout descriptorSetLayout;
    {
        // Configure binding location 0 of this (and the only) descriptor set.
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding0{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        };
        // Use binding information to define the layout of the descriptor set.
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &descriptorSetLayoutBinding0
        };
        auto result = vkCreateDescriptorSetLayout(context, &descriptorSetLayoutInfo, nullptr, &descriptorSetLayout);
        if (result != VK_SUCCESS) {
            std::cerr << std::format("Failed to create descriptor set layout!: {}\n", string_VkResult(result));
        }
    }


    /*
    * ==============================================================================================
    * Create a descriptor pool. 
    * Allocate the necessary descriptor sets from it.
    * Update descriptor sets with their resources.
    * ==============================================================================================
    */
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    {
        // First we provide the pool with info about what resources will be used, and how many of each.
        // For now (ch. 1-7), we only use one type of resource (SSBO). 
        // Over all sets, the max number of resources of this type we will ever need at once is 1.
        VkDescriptorPoolSize SSBOPoolSize{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1
        };
        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, // At most, we will only use one set at once.
            .poolSizeCount = 1, // We created one pool for the SSBO resource type.
            .pPoolSizes = &SSBOPoolSize
        };
        auto result = vkCreateDescriptorPool(context, &descriptorPoolCreateInfo, nullptr, &descriptorPool);
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
        result = vkAllocateDescriptorSets(context, &descriptorSetAllocateInfo, &descriptorSet);
        if (result != VK_SUCCESS) {
            std::cerr << std::format("Failed to allocate descriptor set!: {}\n", string_VkResult(result));
        }

        // Update descriptors with resources.

        VkDescriptorBufferInfo descriptorBufferInfo{
            .buffer = buffer,    // The VkBuffer object
            .offset = 0,
            .range = ssboSizeBytes  // The length of memory to bind; offset is 0.
        };
        VkWriteDescriptorSet descriptorWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &descriptorBufferInfo
        };
        vkUpdateDescriptorSets(
            context,
            1, &descriptorWrite,  // An array of VkWriteDescriptorSet objects
            0, nullptr);          // An array of VkCopyDescriptorSet objects (unused)
    }

    /*
    * ==============================================================================================
    * Create the compute pipeline for the app.
    * ==============================================================================================
    */
    VkPipeline computePipeline;
    VkPipelineLayout computePipelineLayout;
    {
        // Layout for the pipeline.
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
            .pushConstantRangeCount = 0
        };
        auto result = vkCreatePipelineLayout(context, &pipelineLayoutCreateInfo, VK_NULL_HANDLE, &computePipelineLayout);
        if (result != VK_SUCCESS) {
            std::cerr << std::format("Failed to create compute pipeline layout!: {}\n", string_VkResult(result));
        }
        // Create the pipeline.
        VkComputePipelineCreateInfo pipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = computeShaderStageCreateInfo,
            .layout = computePipelineLayout
        };
        vkCreateComputePipelines(context,       // Device
            VK_NULL_HANDLE,                     // Pipeline cache (uses default)
            1, &pipelineCreateInfo,             // Compute pipeline create info
            VK_NULL_HANDLE,                     // Allocator (uses default)
            &computePipeline
        );
    }
    /*
    * ==============================================================================================
    * Allocate a command buffer from the command  pool.
    * This command buffer records the compute dispatch.
    * ==============================================================================================
    */

    // Create and start recording a command buffer
    VkCommandBuffer commandBuffer = AllocateAndBeginOneTimeCommandBuffer(context, commandPool);

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

    EndSubmitWaitAndFreeCommandBuffer(context, context.mQueue, commandPool, commandBuffer);


    /*
    * ==============================================================================================
    * Write SSBO data to an external image.
    * ==============================================================================================
    */
    void* pData;
    vmaMapMemory(allocator, bufferAllocation, &pData);
    stbi_write_hdr("out.hdr", kImageWidth, kImageHeight, 3, reinterpret_cast<float*>(pData));
    vmaUnmapMemory(allocator, bufferAllocation);


    /*
    * ==============================================================================================
    * Clean up Vulkan resources.
    * ==============================================================================================
    */
    vkDestroyPipeline(context, computePipeline, nullptr);
    vkDestroyShaderModule(context, computeShaderModule, nullptr);
    vkDestroyPipelineLayout(context, computePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(context, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(context, descriptorPool, nullptr);
    vmaDestroyBuffer(allocator, indexBuffer, indexBufferAllocation); 
    vmaDestroyBuffer(allocator, vertexBuffer, vertexBufferAllocation);
    vkDestroyCommandPool(context, commandPool, nullptr);
    vmaDestroyBuffer(allocator, buffer, bufferAllocation); //Destroys buffer and frees associated memory.
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
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();
    if (!inst_ret) {
        std::cerr << std::format("Failed to create Vulkan instance: {}\n", inst_ret.error().message());
    }
    vkb::Instance vkb_instance = inst_ret.value();


    // Require these optional features from 1.2.
    VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.bufferDeviceAddress = true;

    // Require these features for the extensions this app uses.
    VkPhysicalDeviceAccelerationStructureFeaturesKHR    asFeatures      { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
    VkPhysicalDeviceRayQueryFeaturesKHR                 rayQueryFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };

    // Select a physical device that supports the required extensions
    // Note that we don't need a surface for this project. 
    vkb::PhysicalDeviceSelector physDeviceSelector{ vkb_instance };
    const auto physDevice_ret = physDeviceSelector
        .set_minimum_version(1, 2)
        .defer_surface_initialization()
        .set_required_features_12(features12) 
        .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        .add_required_extension_features(asFeatures)
        .add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
        .add_required_extension_features(rayQueryFeatures)
        .select();
    if (!physDevice_ret) {
        std::cerr << std::format("Failed to create Vulkan physical device: {}\n", physDevice_ret.error().message());
    }
    vkb::PhysicalDevice vkb_physical_device = physDevice_ret.value();

    // Create logical device.
    vkb::DeviceBuilder device_builder(vkb_physical_device);
    const auto dev_ret = device_builder.build();
    if (!dev_ret) {
        std::cerr << std::format("Failed to create Vulkan logical device: {}\n", dev_ret.error().message());
    }
    vkb::Device vkb_device = dev_ret.value();

    // Search device for a compute queue.
    const auto queue_ret = vkb_device.get_queue(vkb::QueueType::compute);
    if (!queue_ret) {
        std::cerr << std::format("Failed to find a compute queue: {}\n", queue_ret.error().message());
    }
    VkQueue compute_queue = queue_ret.value();
    const auto compute_queue_index = vkb_device.get_queue_index(vkb::QueueType::compute).value();

    return { vkb_instance,vkb_physical_device,vkb_device, {compute_queue, compute_queue_index } };
}



VmaAllocator initAllocator(VulkanContextData& context)
{
    // Create memory allocator.
    VmaAllocatorCreateInfo allocatorInfo {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = context.mPhysicalDevice,
        .device = context.mDevice,
        .instance = context.mInstance
    };
    VmaAllocator allocator;
    auto result = vmaCreateAllocator(&allocatorInfo, &allocator);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create VmaAllocator: {}\n", string_VkResult(result));
    }
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

VkShaderModule createShaderModule(VkDevice device, const std::string& path)
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
    auto result = vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create shader module: {}\n", string_VkResult(result));
    }
    return shaderModule;
}


VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool)
{
    //Allocate a command buffer from the pool.
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = cmdPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VkCommandBuffer commandBuffer;
    auto result = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to create Command buffer: {}\n", string_VkResult(result));
    }

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT // After submitting it we will re-record it before submitting again.
    };
    result = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to begin recording Command buffer: {}\n", string_VkResult(result));
    }
    return commandBuffer;
}

// Ends recording of a commmand buffer.
// Then submits it to a queue.
// Then waits for the queue to finish processing commands, 
// Finally, destroys the command buffer.
void EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer)
{
    // End recording of command buffer
    auto result = vkEndCommandBuffer(cmdBuffer);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to end recording Command buffer: {}\n", string_VkResult(result));
    }


    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuffer
    };
    result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to submit Command buffer to queue: {}\n", string_VkResult(result));
    }

    // Wait until queue has no more work to do.
    // Note that it is not enough alone. This is why we inserted a pipeline barrier.
    result = vkQueueWaitIdle(queue);
    if (result != VK_SUCCESS) {
        std::cerr << std::format("Failed to finish processing compute queue: {}\n", string_VkResult(result));
    }

    vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}


VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer)
{
    VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer
    };
    return vkGetBufferDeviceAddress(device, &addressInfo);
}

MeshData LoadMeshFromObj(const std::string& path)
{

    std::string inputfile = "scenes/cornell_box.obj";
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    // Get vertices of the mesh.
    const std::vector<tinyobj::real_t>   objVertices = reader.GetAttrib().GetVertices();

    const std::vector<tinyobj::shape_t>& objShapes = reader.GetShapes();  // All shapes in the file
    assert(objShapes.size() == 1);                                          // Check that this file has only one shape
    const tinyobj::shape_t& objShape = objShapes[0];                        // Get the first shape

    // Get the indices of the vertices of the first mesh of `objShape` in `attrib.vertices`:
    std::vector<uint32_t> objIndices;
    objIndices.reserve(objShape.mesh.indices.size());
    for (const tinyobj::index_t& index : objShape.mesh.indices)
    {
        objIndices.push_back(index.vertex_index);
    }

    return { objVertices, objIndices };
}