cmake_minimum_required(VERSION 4.0)
project(cudaPlayground CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cudaPlayground main.cu
        error_checks.h
        main.h)

set_target_properties(cudaPlayground PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
