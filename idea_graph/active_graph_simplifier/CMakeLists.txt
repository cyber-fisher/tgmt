cmake_minimum_required(VERSION 3.16)
project(ActiveGraphMergeTreeSimplifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(VTK REQUIRED)

if(TARGET ttkMergeTree)
  add_executable(${PROJECT_NAME}
    main.cpp
    ActiveGraphMergeTreeSimplifier.cpp
    ActiveGraphPostProcess.cpp
    MergeTreeIO.cpp
  )

  target_link_libraries(${PROJECT_NAME}
    PRIVATE
      ttkMergeTree
      ${VTK_LIBRARIES}
  )
else()
  add_executable(${PROJECT_NAME}
    main.cpp
    ActiveGraphMergeTreeSimplifier.cpp
    ActiveGraphPostProcess.cpp
    MergeTreeIO.cpp
  )

  target_link_libraries(${PROJECT_NAME}
    PRIVATE
      ${VTK_LIBRARIES}
  )
endif()

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
    INSTALL_RPATH
      "${CMAKE_INSTALL_RPATH}"
)