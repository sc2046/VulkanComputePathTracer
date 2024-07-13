//#define GLFW_INCLUDE_VULKAN
//#include <GLFW/glfw3.h>

//#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>

#include <format>
#include <iostream>

#include <vulkan/vulkan.h>
#include <VkBootstrap.h>


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
    VkPhysicalDevice physical_device = vkb_physical_device.physical_device;

    std::cout << "Vulkan physical device selected successfully!\n";

    // Create logical device.
    vkb::DeviceBuilder device_builder(vkb_physical_device);
    auto dev_ret = device_builder
        .build();
    if (!dev_ret) {
        std::cerr << std::format("Failed to create Vulkan logical device: {}\n", dev_ret.error().message());
    }
    vkb::Device vkb_device = dev_ret.value();
    VkDevice vDevice = vkb_device.device;
    
    std::cout << "Vulkan logical device created successfully!\n";


    // Get queue from device
    auto queue_ret = vkb_device.get_queue(vkb::QueueType::compute);
    if (!queue_ret) {
        std::cerr << std::format("Failed to find a compute queue: {}\n", queue_ret.error().message());
    }
    VkQueue compute_queue = queue_ret.value();
    std::cout << "Compute queue found!\n";







    vkb::destroy_device(vkb_device);
    vkb::destroy_instance(vkb_instance);
}
