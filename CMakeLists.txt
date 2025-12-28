cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-fno-omit-frame-pointer)
add_link_options(-fno-omit-frame-pointer)

project(canny_lane_tracker)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} header)

add_executable(canny_lane_tracker src/main.cpp src/video_service.cpp src/canny_edge_detection.cpp)

target_link_libraries(canny_lane_tracker ${OpenCV_LIBS} yaml-cpp)

target_compile_options(canny_lane_tracker PRIVATE 
    $<$<CONFIG:Release>:-O3 >
    $<$<CONFIG:RelWithDebInfo>:-O3 -g>
    $<$<CONFIG:Debug>:O0 -g>
    -fno-omit-frame-pointer
)
target_link_options(canny_lane_tracker PRIVATE 
    -fno-omit-frame-pointer
)