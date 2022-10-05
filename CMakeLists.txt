cmake_minimum_required(VERSION 3.18)

project("foray-denoiser-optix")

enable_language(CXX)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)

include("cmakescripts/findoptixsdk.cmake")

MESSAGE("--- << CMAKE of ${PROJECT_NAME} >> --- ")
MESSAGE(STATUS "CURRENT SOURCE DIR \"${CMAKE_CURRENT_SOURCE_DIR}\"")#

# Enable strict mode for own code
SET(CMAKE_CXX_FLAGS ${STRICT_FLAGS})

# collect sources
file(GLOB_RECURSE src "src/*.cpp")
add_library(${PROJECT_NAME} ${src})
set_target_properties(${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME}
                       PROPERTIES POSITION_INDEPENDENT_CODE ON)

# find vulkan
find_package(Vulkan REQUIRED)

target_link_libraries(
	${PROJECT_NAME}
	PUBLIC rtrpf
	PUBLIC cuda
	PUBLIC cudart
)

if (WIN32)
	target_link_libraries(
		${PROJECT_NAME}
		PUBLIC ${SDL2_LIBRARIES}
	)
endif()

target_include_directories(
	${PROJECT_NAME}
	PUBLIC "../hsk_rt_rpf/src"
	PUBLIC "../hsk_rt_rpf/third_party"
	PUBLIC $CACHE{OPTIXSDK_INCLUDE}
	PUBLIC ${Vulkan_INCLUDE_DIRS}
)
