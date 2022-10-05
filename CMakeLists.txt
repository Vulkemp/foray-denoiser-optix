cmake_minimum_required(VERSION 3.18)

project("hsk_sponza_sample")

# Include Compiler Config (sets c++ 20 and compiler flags)
include("hsk_rt_rpf/cmakescripts/compilerconfig.cmake")

# Run Locate SDL2 script
set(SDL2_HINT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hsk_rt_rpf/third_party")
include("hsk_rt_rpf/cmakescripts/locatesdl2.cmake")


# Add subdirectories
if (NOT TARGET "rtrpf")
    add_subdirectory("hsk_rt_rpf")
endif()
add_subdirectory("src")