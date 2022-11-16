cmake_minimum_required(VERSION 3.20)

project("foray-denoiser-optix")

enable_language(CXX)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)

include("cmakescripts/findoptixsdk.cmake")
include("cmakescripts/compileshader.cmake")

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
	PUBLIC foray
)

if (WIN32)
	set(CUDA_LIB "$ENV{CUDA_PATH}/lib/x64/cuda.lib")
	set(CUDART_LIB "$ENV{CUDA_PATH}/lib/x64/cudart.lib")

	target_link_libraries(
		${PROJECT_NAME}
		PUBLIC ${SDL2_LIBRARIES}
		PUBLIC ${CUDART_LIB}
		PUBLIC ${CUDA_LIB}
	)
else ()
	target_link_libraries(
		${PROJECT_NAME}
		PUBLIC cuda
		PUBLIC cudart
)

endif()

foray_compileshader("${CMAKE_CURRENT_SOURCE_DIR}/src/shader/scalemotion.comp" "${CMAKE_CURRENT_SOURCE_DIR}/src/foray_optix_scalemotion.comp.spv.h")

target_include_directories(
	${PROJECT_NAME}
	PUBLIC "${CMAKE_SOURCE_DIR}/foray/src"
	PUBLIC "${CMAKE_SOURCE_DIR}/foray/third_party"
	PUBLIC $CACHE{OPTIXSDK_INCLUDE}
	PUBLIC ${Vulkan_INCLUDE_DIRS}
	PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
