PROJECT( helloworld_proj )
FIND_PACKAGE( OpenCV REQUIRED )


ADD_EXECUTABLE(show-camera 
	show-camera.cpp 
        face_recognation.h
	face_recognation.cpp
	)

add_subdirectory(dlib dlib_build)

TARGET_LINK_LIBRARIES(show-camera ${OpenCV_LIBS} dlib::dlib)
cmake_minimum_required(VERSION 3.5)
