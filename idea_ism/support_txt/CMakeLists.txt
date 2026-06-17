cmake_minimum_required(VERSION 3.21)

project(treeTxtIsmSimplifier LANGUAGES CXX)

find_package(VTK REQUIRED)

add_executable(treeTxtIsmSimplifier
  TreeTxtIsmSimplifier.cpp
)

target_link_libraries(treeTxtIsmSimplifier
  PRIVATE
    ${VTK_LIBRARIES}
)

target_compile_features(treeTxtIsmSimplifier PRIVATE cxx_std_17)
