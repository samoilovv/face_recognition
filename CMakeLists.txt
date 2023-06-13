PROJECT( helloworld_proj )
FIND_PACKAGE( OpenCV REQUIRED )

#set(dlib_DIR "c:/build-dlib-Desktop_Qt_5_15_2_MSVC2019_64bit-Debug")
#find_package(dlib REQUIRED)

ADD_EXECUTABLE( show-camera show-camera.cpp )

add_subdirectory("../../../dlib/dlib" dlib_build)

TARGET_LINK_LIBRARIES( show-camera ${OpenCV_LIBS} dlib::dlib)
cmake_minimum_required(VERSION 2.8)
