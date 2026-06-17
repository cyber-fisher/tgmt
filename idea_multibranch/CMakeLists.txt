cmake_minimum_required(VERSION 3.16)
project(TreeTxtMultiBranchSimplifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(VTK REQUIRED)

add_executable(treeTxtMultiBranchSimplifier
  TreeTxtMultiBranchSimplifier.cpp
)

add_executable(treeTxtMultiBranchVolumeSimplifier
  TreeTxtMultiBranchVolumeSimplifier.cpp
)

add_executable(treeTxtSaddleVolumeWeightedSimplifier
  TreeTxtSaddleVolumeWeightedSimplifier.cpp
)

target_link_libraries(treeTxtMultiBranchSimplifier
  PRIVATE
    ${VTK_LIBRARIES}
)

target_link_libraries(treeTxtMultiBranchVolumeSimplifier
  PRIVATE
    ${VTK_LIBRARIES}
)

target_link_libraries(treeTxtSaddleVolumeWeightedSimplifier
  PRIVATE
    ${VTK_LIBRARIES}
)

target_include_directories(treeTxtMultiBranchSimplifier
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(treeTxtMultiBranchVolumeSimplifier
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(treeTxtSaddleVolumeWeightedSimplifier
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
