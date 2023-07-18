cmake_minimum_required(VERSION 3.5)

project(face_recognation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/opencv/include
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/opencv/modules/calib3d/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/core/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudaarithm/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudabgsegm/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudacodec/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudafeatures2d/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudafilters/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudaimgproc/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudalegacy/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudaobjdetect/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudaoptflow/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudastereo/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudawarping/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/cudev/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/dnn/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/features2d/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/flann/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/highgui/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/imgcodecs/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/imgproc/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/ml/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/objdetect/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/photo/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/shape/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/stitching/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/superres/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/ts/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/video/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/videoio/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/videostab/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/viz/include
    ${CMAKE_SOURCE_DIR}/opencv/modules/world/include
)

add_executable(${PROJECT_NAME}
    main.cpp
)

add_subdirectory(opencv opencv_build)
set(JPEG_FOUND 0)

add_subdirectory(dlib dlib_build)



target_link_libraries(${PROJECT_NAME}
#    ${OpenCV_LIBS}
opencv_calib3d
opencv_core
opencv_dnn
opencv_features2d
opencv_flann
opencv_gapi
opencv_highgui
opencv_imgcodecs
opencv_imgproc
opencv_ml
opencv_objdetect
opencv_photo
opencv_stitching
opencv_video
opencv_videoio
dlib::dlib
)

set(landmarks5 shape_predictor_5_face_landmarks.dat)
set(model dlib_face_recognition_resnet_model_v1.dat)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${landmarks5} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${model} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/face_samples)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/face_samples/face_sample.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/face_samples)
