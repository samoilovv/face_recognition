cmake_minimum_required(VERSION 3.5)

project(face_recognation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
    show-camera.cpp
    face_recognation.h
    face_recognation.cpp
)

add_subdirectory(dlib dlib_build)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    dlib::dlib
)

