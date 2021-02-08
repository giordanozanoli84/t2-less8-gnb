cmake_minimum_required(VERSION 3.13)
project(gaussian_naive_bayes)

set(CMAKE_CXX_STANDARD 11)

if (WIN32)
    include_directories(C:/Users/lei00017/Documents/cpplibs/eigen)
endif (WIN32)
if (UNIX)
    include_directories(/usr/include/eigen3)
endif (UNIX)

file(COPY ./data DESTINATION "${CMAKE_BINARY_DIR}")

add_executable(gaussian_naive_bayes main.cpp gnb.cpp gnb.h helpers.cpp helpers.h)