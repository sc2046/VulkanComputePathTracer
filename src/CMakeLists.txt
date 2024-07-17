

# Add source to this project's executable.
add_executable (VulkanComputePathTracer "main.cpp"   "tiny_obj_loader.h")

add_custom_command(
	TARGET VulkanComputePathTracer POST_BUILD 
	COMMAND ${CMAKE_COMMAND} -E copy_directory #recursively copy
	${CMAKE_SOURCE_DIR}/src/shaders 
	$<TARGET_FILE_DIR:VulkanComputePathTracer>/shaders)
set_property(TARGET VulkanComputePathTracer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/$(Configuration)")

target_link_libraries(VulkanComputePathTracer glfw)
target_link_libraries(VulkanComputePathTracer vk-bootstrap)
target_link_libraries(VulkanComputePathTracer ${Vulkan_LIBRARY})
# Link Vulkan library
#target_link_libraries(VulkanComputePathTracer Vulkan::Vulkan)




