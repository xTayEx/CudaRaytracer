cmake_minimum_required(VERSION 3.10)
project(RaytracerCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(raytracer_cuda main.cu)
set_target_properties(raytracer_cuda PROPERTIES CUDA_ARCHITECTURES "89")
