cmake_minimum_required(VERSION 3.10)
project(RaytracerCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp")
add_executable(raytracer_cuda main.cu ${SOURCES})
target_include_directories(raytracer_cuda PRIVATE include)
set_target_properties(raytracer_cuda PROPERTIES CUDA_ARCHITECTURES "89")
