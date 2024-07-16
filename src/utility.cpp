#include <VkBootstrap.h>
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

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
}