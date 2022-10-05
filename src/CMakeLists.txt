cmake_minimum_required(VERSION 3.18)

project("foray-denoiser-optix")

MESSAGE("--- << CMAKE of ${PROJECT_NAME} >> --- ")
MESSAGE(STATUS "CURRENT SOURCE DIR \"${CMAKE_CURRENT_SOURCE_DIR}\"")#

# Enable strict mode for own code
SET(CMAKE_CXX_FLAGS ${STRICT_FLAGS})

# collect sources
file(GLOB_RECURSE src "*.cpp")
add_library(${PROJECT_NAME} ${src})

# find vulkan
find_package(Vulkan REQUIRED)

target_link_libraries(
	${PROJECT_NAME}
	PUBLIC rtrpf
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
	PUBLIC ${Vulkan_INCLUDE_DIRS}
)
