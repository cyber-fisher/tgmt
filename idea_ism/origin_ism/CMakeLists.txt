cmake_minimum_required(VERSION 3.21)

project(originIsmTreeSimplifier LANGUAGES CXX)

find_package(VTK REQUIRED)

add_executable(originIsmTreeSimplifier
  OriginIsmTreeSimplifier.cpp
)

target_link_libraries(originIsmTreeSimplifier
  PRIVATE
    ${VTK_LIBRARIES}
)

target_compile_features(originIsmTreeSimplifier PRIVATE cxx_std_17)
