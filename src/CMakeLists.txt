

# Add source to this project's executable.
add_executable (VulkanComputePathTracer "main.cpp")


target_link_libraries(VulkanComputePathTracer glfw)
target_link_libraries(VulkanComputePathTracer vk-bootstrap)
target_link_libraries(VulkanComputePathTracer ${Vulkan_LIBRARY})
# Link Vulkan library
#target_link_libraries(VulkanComputePathTracer Vulkan::Vulkan)




