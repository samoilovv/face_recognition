cmake_minimum_required(VERSION 3.5)

project(face_recognation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
    main.cpp
)

add_subdirectory(dlib dlib_build)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    dlib::dlib
)

set(landmarks5 shape_predictor_5_face_landmarks.dat)
set(model dlib_face_recognition_resnet_model_v1.dat)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${landmarks5} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${model} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/face_samples)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/face_samples/face_sample.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/face_samples)
