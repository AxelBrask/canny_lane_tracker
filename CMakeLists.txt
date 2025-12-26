cmake_minimum_required(VERSION 3.10)

project(canny_lane_tracker)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} header)

add_executable(canny_lane_tracker src/main.cpp src/video_service.cpp src/canny_edge_detection.cpp)

target_link_libraries(canny_lane_tracker ${OpenCV_LIBS} yaml-cpp)