set(OPTIXSDK_INCLUDE CACHE PATH "Include directory path of OptiX SDK")

if (NOT EXISTS $CACHE{OPTIXSDK_INCLUDE})
    message(FATAL_ERROR "Set the cache entry OPTIXSDK_INCLUDE (\"$CACHE{OPTIXSDK_INCLUDE}\") to the include directory of the OptiX SDK")
else ()
    message("Using \"$CACHE{OPTIXSDK_INCLUDE}\" for OptiX include")
endif ()