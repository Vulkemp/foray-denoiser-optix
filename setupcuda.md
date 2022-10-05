# Assure Build Tools and Cuda are on compatible versions

1. Make sure Cuda dev tools are installed and not out of date
1. Make sure Cuda has an entry in PATH (if windows) or it has set CUDA_CXX env variable
1. Make sure you target an x64 build

# Configure CMake Cache

1. Edit `OPTIXSDK_INCLUDE` path in CMakeCache.txt to be an absolute path pointing to OptiX SDK include folder