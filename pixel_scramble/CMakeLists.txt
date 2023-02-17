cmake_minimum_required(VERSION 3.0.0)
project(pixel_scramble VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(pixel_scramble main.cpp)

# needed to find and link opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(pixel_scramble ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
