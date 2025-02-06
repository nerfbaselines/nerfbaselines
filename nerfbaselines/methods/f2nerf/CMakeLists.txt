cmake_minimum_required(VERSION 3.18)
project(f2nerf
        VERSION 0.1
        LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

###############################################################################
# CUDA compiler setup
###############################################################################
find_package(Python REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=20236")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_definitions(-DHALF_PRECISION)

set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

# zlib, for compiling cnpy
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# tiny-cuda-nn
add_definitions(-DJSON_HAS_CPP_14)
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(External/tiny-cuda-nn)
include_directories(${TCNN_INCLUDE_DIRECTORIES})
include_directories(External/tiny-cuda-nn/include)
include_directories(External/tiny-cuda-nn/dependencies)

# YAML
add_subdirectory(External/yaml-cpp)
include_directories(External/yaml-cpp/include)

# eigen
include_directories(External/eigen-3.4.0)

include(src/Field/field.cmake)
include(src/Dataset/dataset.cmake)
include(src/PtsSampler/pts_sampler.cmake)
include(src/Renderer/renderer.cmake)
include(src/Shader/shader.cmake)
include(src/Utils/utils.cmake)

list(APPEND SRC_ALL
        ${SRC_FIELD}
        ${SRC_DATASET}
        ${SRC_GEOMETRY}
        ${SRC_PTS_SAMPLER}
        ${SRC_RENDERER}
        ${SRC_SHADER}
        ${SRC_UTILS})
list(APPEND SRC_ALL src/ExpRunner.cpp)

set(ALL_LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${ZLIB_LIBRARIES} yaml-cpp)
set(ALL_LIBS ${ALL_LIBS} stdc++fs Python::Python fmt::fmt-header-only)

# Remove fmt from tiny-cuda-nn and add ALL_LIBS
get_target_property(_libs tiny-cuda-nn INTERFACE_LINK_LIBRARIES)
list(REMOVE_ITEM _libs fmt)
set_property(TARGET tiny-cuda-nn PROPERTY INTERFACE_LINK_LIBRARIES ${_libs})
target_link_libraries(tiny-cuda-nn PUBLIC ${ALL_LIBS})

pybind11_add_module(f2nerf ${SRC_ALL} extension.cpp)
target_link_libraries(f2nerf PUBLIC ${ALL_LIBS} tiny-cuda-nn)
set_property(TARGET f2nerf PROPERTY CXX_STANDARD 17)
set_property(TARGET f2nerf PROPERTY CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
