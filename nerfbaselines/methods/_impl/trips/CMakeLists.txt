cmake_minimum_required (VERSION 3.9)

project(trips LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

add_library(trips SHARED
  trips.cu
)
target_compile_features(trips PRIVATE cxx_std_11)
target_link_libraries(trips PRIVATE ${TORCH_LIBRARIES} Python::Python)