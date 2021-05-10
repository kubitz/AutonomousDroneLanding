cmake_minimum_required(VERSION 3.19)
project(FYPLanding)
set(CMAKE_CXX_STANDARD 17)
FIND_PACKAGE( OpenCV REQUIRED )

add_executable(FYPLanding main.cpp)
TARGET_LINK_LIBRARIES (FYPLanding ${OpenCV_LIBS})