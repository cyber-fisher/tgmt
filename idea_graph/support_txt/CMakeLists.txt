cmake_minimum_required(VERSION 3.16)
project(TreeTxtActiveGraphSimplifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(VTK REQUIRED)

add_executable(treeTxtActiveGraphSimplifier
  TreeTxtActiveGraphSimplifier.cpp
)

target_link_libraries(treeTxtActiveGraphSimplifier
  PRIVATE
    ${VTK_LIBRARIES}
)

target_include_directories(treeTxtActiveGraphSimplifier
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
