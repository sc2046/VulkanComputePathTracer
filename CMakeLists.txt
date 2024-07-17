# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project(
  "VulkanComputePathTracer"
  VERSION 1.0
  LANGUAGES C CXX
)

# ============================================================================
# compile using C++20
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_BINARY_DIR "${PROJECT_SOURCE_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib/$<CONFIG>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib/$<CONFIG>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/$<CONFIG>")

set(GLFW_INCLUDE_DIRS  "${PROJECT_SOURCE_DIR}/external/glfw/include")
set(GLM_INCLUDE_DIRS   "${PROJECT_SOURCE_DIR}/external/glm/glm")
set(STB_INCLUDE_DIRS   "${PROJECT_SOURCE_DIR}/external/stb")
set(VKBOOTSTRAP_INCLUDE_DIRS   "${PROJECT_SOURCE_DIR}/external/vk-bootstrap")
set(VMA_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include")
set(VOLD_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/volk/include")


find_package(Vulkan REQUIRED)

include_directories(
    ${GLFW_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
    ${STB_INCLUDE_DIRS}
    ${VKBOOTSTRAP_INCLUDE_DIRS}
    ${VMA_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIR}
)

add_subdirectory("external/glfw")  
add_subdirectory("external/volk")  
add_subdirectory("external/vk-bootstrap")
add_subdirectory("src")
