cmake_minimum_required(VERSION 3.10)
project(FYPLanding)
set(CMAKE_CXX_STANDARD 17)
FIND_PACKAGE(OpenCV REQUIRED)
set(CMAKE_PREFIX_PATH /home/kubitz/.local/lib/python3.6/site-packages/torch)


list(APPEND CMAKE_PREFIX_PATH "/path/to/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message(STATUS "Torch path. {${TORCH_INSTALL_PREFIX}}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
link_libraries(stdc++fs)

add_executable(FYPLanding
        src/main.cpp
        src/LzFinder.cpp
        include/LzFinder.h
        include/labels.h
        src/labels.cpp
        include/labels.h
        src/utils.cpp
        include/utils.h
        src/ObjectDetector.cpp
        include/ObjectDetector.h
        src/Segmentator.cpp
        include/Segmentator.h
        )

TARGET_LINK_LIBRARIES(FYPLanding ${OpenCV_LIBS} ${TORCH_LIBRARIES})