//#define GLFW_INCLUDE_VULKAN
//#include <GLFW/glfw3.h>

//#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>

#include <format>
#include <iostream>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <VkBootstrap.h>
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"


static constexpr uint32_t kImageWidth{ 800 };
static constexpr uint32_t kImageHeight{ 600 };


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
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice    = physicalDevice;
    allocatorInfo.device            = device;
    allocatorInfo.instance          = instance;
    VmaAllocator allocator;
    VkResult result = vmaCreateAllocator(&allocatorInfo, &allocator);
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
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.requiredFlags = 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT     | // Specify that the buffer may be mapped. 
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT    | // 
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;       // Without this flag, every read of the buffer's memory requires a fetch from GPU memory!
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    // Create a buffer.
    VkBuffer buffer;
    VmaAllocation bufferAllocation;
    vmaCreateBuffer(allocator, &bufferCreateInfo, &allocInfo, &buffer, &bufferAllocation, nullptr);

    // Map data from GPU to CPU to read it.
    void* mappedData;
    vmaMapMemory(allocator, bufferAllocation, &mappedData);
    float* fltData = reinterpret_cast<float*>(mappedData);
    printf("First three elements: %f, %f, %f\n", fltData[0], fltData[1], fltData[2]);
    vmaUnmapMemory(allocator, bufferAllocation);





    vmaDestroyBuffer(allocator, buffer, bufferAllocation);
    vmaDestroyAllocator(allocator);
    vkb::destroy_device(vkb_device);
    vkb::destroy_instance(vkb_instance);
}
