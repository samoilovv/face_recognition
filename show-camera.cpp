/*
 #include <opencv2/opencv.hpp>
#include <iostream>

int main(int, char**) {
    // open the first webcam plugged in the computer
    cv::VideoCapture camera(0); // in linux check $ ls /dev/video0
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    // create a window to display the images from the webcam
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

    // array to hold image
    cv::Mat frame;

    // display the frame until you press a key
    while (1) {
        // capture the next frame from the webcam
        camera >> frame;
        // show the image on the window
        cv::imshow("Webcam", frame);
        // wait (10ms) for esc key to be pressed to stop
        if (cv::waitKey(10) == 27)
            break;
    }
    return 0;
}


// CMD to generate executable:
// g++ webcam_opencv.cpp -o webcam_demo -I/usr/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui

// Note: check your opencv hpp files - for many users it is at /usr/local/include/opencv4
// Add more packages during compilation from the list obtained by $ pkg-config --cflags --libs opencv4

*/

//#include <iostream>
//#include <vector>
//#include <string>
//#include <opencv2/opencv.hpp>
//#include <dlib/opencv.h>
//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>

//using namespace std;
//using namespace cv;
//using namespace dlib;

//int main()
//{
//    // Load face detection model
//    frontal_face_detector faceDetector = get_frontal_face_detector();

//    // Load face recognition model
//    shape_predictor shapePredictor;
//    deserialize("shape_predictor_68_face_landmarks.dat") >> shapePredictor;
//    anet_type net;
//    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

//    // Load known face encodings and names
//    std::vector<matrix<float, 0, 1>> known_face_encodings;
//    std::vector<std::string> known_face_names;
//    matrix<float, 0, 1> obama_face_encoding;
//    matrix<float, 0, 1> biden_face_encoding;

//    // Load and encode sample images
//    array2d<rgb_pixel> obama_image;
//    load_image(obama_image, "samoilov.jpg");
//    std::vector<dlib::rectangle> obama_face_rects = faceDetector(obama_image);
//    matrix<rgb_pixel> obama_face_chip;
//    extract_image_chip(obama_image, get_face_chip_details(obama_face_rects[0], 150, 0.25), obama_face_chip);
//    obama_face_encoding = net(obama_face_chip);
//    known_face_encodings.push_back(obama_face_encoding);
//    known_face_names.push_back("Vladimir");

//    array2d<rgb_pixel> biden_image;
//    load_image(biden_image, "biden.jpg");
//    std::vector<dlib::rectangle> biden_face_rects = faceDetector(biden_image);
//    matrix<rgb_pixel> biden_face_chip;
//    extract_image_chip(biden_image, get_face_chip_details(biden_face_rects[0], 150, 0.25), biden_face_chip);
//    biden_face_encoding = net(biden_face_chip);
//    known_face_encodings.push_back(biden_face_encoding);
//    known_face_names.push_back("Joe Biden");

//    // Initialize variables
//    std::vector<dlib::rectangle> face_rects;
//    std::vector<matrix<float, 0, 1>> face_encodings;
//    std::vector<std::string> face_names;
//    bool process_this_frame = true;

//    // Open video capture
//    cv::VideoCapture video_capture(0);
//    if (!video_capture.isOpened())
//    {
//        std::cerr << "Error opening video capture." << std::endl;
//        return -1;
//    }

//    // Create named window
//    cv::namedWindow("Video", cv::WINDOW_NORMAL);

//    while (true)
//    {
//        // Grab a single frame from video capture
//        cv::Mat frame;
//        video_capture >> frame;

//        // Convert frame to dlib's image format
//        cv_image<rgb_pixel> dlib_frame(frame);

//        // Process every other frame to save time
//        if (process_this_frame)
//        {
//            // Detect faces in the frame
//            face_rects = faceDetector(dlib_frame);

//            // Iterate over detected faces
//            // Encode face encodings for the detected faces
//            face_encodings.clear();
//            for (const auto& face_rect : face_rects)
//            {
//                auto shape = shapePredictor(dlib_frame, face_rect);
//                matrix<rgb_pixel> face_chip;
//                extract_image_chip(dlib_frame, get_face_chip_details(shape, 150, 0.25), face_chip);
//                face_encodings.push_back(net(face_chip));
//            }

//            // Match faces with known faces
//            face_names.clear();
//            for (const auto& face_encoding : face_encodings)
//            {
//                std::vector<bool> matches;
//                for (const auto& known_face_encoding : known_face_encodings)
//                {
//                    matches.push_back(dlib::length(face_encoding - known_face_encoding) < 0.6);
//                }

//                std::string name = "Unknown";
//                auto best_match = std::min_element(matches.begin(), matches.end());
//                if (*best_match)
//                {
//                    auto index = std::distance(matches.begin(), best_match);
//                    name = known_face_names[index];
//                }

//                face_names.push_back(name);
//            }
//        }

//        process_this_frame = !process_this_frame;

//        // Display the results
//        for (size_t i = 0; i < face_rects.size(); ++i)
//        {
//            // Scale back up the face locations since we processed the frame at a lower resolution
//            dlib::rectangle face_rect(
//                face_rects[i].left() * 4,
//                face_rects[i].top() * 4,
//                face_rects[i].right() * 4,
//                face_rects[i].bottom() * 4
//            );

//            // Draw a box around the face
//            cv::rectangle(frame, cv::Point(face_rect.left(), face_rect.top()), cv::Point(face_rect.right(), face_rect.bottom()), cv::Scalar(0, 0, 255), 2);

//            // Draw a label with a name below the face
//            cv::rectangle(frame, cv::Point(face_rect.left(), face_rect.bottom() - 35), cv::Point(face_rect.right(), face_rect.bottom()), cv::Scalar(0, 0, 255), cv::FILLED);
//            cv::putText(frame, face_names[i], cv::Point(face_rect.left() + 6, face_rect.bottom() - 6), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
//        }

//        // Display the resulting frame
//        cv::imshow("Video", frame);

//        // Check for 'q' key press to exit
//        if (cv::waitKey(1) == 'q')
//        {
//            break;
//        }
//    }

//    // Release video capture
//    video_capture.release();

//    // Close all windows
//    cv::destroyAllWindows();

//    return 0;
//}


//#include <iostream>
//#include <vector>
//#include <string>
//#include <opencv2/opencv.hpp>
//#include <dlib/opencv.h>
//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing.h>
//#include <dlib/image_io.h>
//#include <dlib/gui_widgets.h>

//using namespace std;
//using namespace dlib;

//int main()
//{
//    cv::VideoCapture video_capture(0);
//    if (!video_capture.isOpened())
//    {
//        cout << "Failed to open webcam." << endl;
//        return 1;
//    }

//    frontal_face_detector face_detector = get_frontal_face_detector();
//    shape_predictor face_landmark_predictor;
//    deserialize("shape_predictor_68_face_landmarks.dat") >> face_landmark_predictor;

//    matrix<rgb_pixel> obama_image;
//    load_image(obama_image, "samoilov.jpg");
//    matrix<float, 0, 1> obama_face_encoding = face_recognition::face_encodings(obama_image, face_landmark_predictor)[0];



//    matrix<rgb_pixel> biden_image;
//    load_image(biden_image, "biden.jpg");
//    matrix<float, 0, 1> biden_face_encoding = face_recognition::face_encodings(biden_image, face_landmark_predictor)[0];

//    std::vector<matrix<float, 0, 1>> known_face_encodings = {obama_face_encoding, biden_face_encoding};
//    std::vector<string> known_face_names = {"Vladimir", "Joe Biden"};

//    bool process_this_frame = true;
//    cv::Mat frame;

//    while (true)
//    {
//        video_capture.read(frame);
//        if (frame.empty())
//        {
//            break;
//        }

//        if (process_this_frame)
//        {
//            cv_image<bgr_pixel> cimg(frame);
//            matrix<rgb_pixel> img;
//            assign_image(img, cimg);

//            std::vector<matrix<rgb_pixel>> faces;
//            std::vector<rectangle> face_locations = face_detector(img);

//            for (rectangle face_location : face_locations)
//            {
//                auto shape = face_landmark_predictor(img, face_location);
//                matrix<rgb_pixel> face_chip;
//                extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
//                faces.push_back(move(face_chip));
//            }

//            std::vector<matrix<float, 0, 1>> face_encodings = face_recognition::face_encodings(faces);

//            std::vector<string> face_names;
//            for (auto face_encoding : face_encodings)
//            {
//                std::vector<bool> matches = face_recognition::compare_faces(known_face_encodings, face_encoding);
//                string name = "Unknown";

//                float best_distance = numeric_limits<float>::max();
//                int best_match_index = -1;

//                for (int i = 0; i < matches.size(); i++)
//                {
//                    if (matches[i])
//                    {
//                        float distance = face_recognition::face_distance(known_face_encodings[i], face_encoding);
//                        if (distance < best_distance)
//                        {
//                            best_distance = distance;
//                            best_match_index = i;
//                        }
//                    }
//                }

//                if (best_match_index != -1)
//                {
//                    name = known_face_names[best_match_index];
//                }

//                face_names.push_back(name);
//            }

//            int i = 0;
//            for (rectangle face_location : face_locations)
//            {
//                cv::Rect rect(face_location.left(), face_location.top(), face_location.width(), face_location.height());
//                cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);

//                cv::Rect name_rect(face_location.left(), face_location.bottom(), face_location.width(), 35);
//                cv::rectangle(frame, name_rect, cv::Scalar(0, 0, 255), cv::FILLED);

//                cv::putText(frame, face_names[i], cv::Point(face_location.left() + 6, face_location.bottom() - 6), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 255, 255), 1);

//                i++;
//            }
//        }

//        process_this_frame = !process_this_frame;

//        cv::imshow("Video", frame);

//        if (cv::waitKey(1) == 'q')
//        {
//            break;
//        }
//    }

//    video_capture.release();
//    cv::destroyAllWindows();

//    return 0;
//}


#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;

int main()
{
    cv::VideoCapture video_capture(0);
    if (!video_capture.isOpened())
    {
        cout << "Failed to open webcam." << endl;
        return 1;
    }

    frontal_face_detector face_detector = get_frontal_face_detector();
    shape_predictor face_landmark_predictor;
    deserialize("shape_predictor_68_face_landmarks.dat") >> face_landmark_predictor;

    cv::Mat obama_image = cv::imread("samoilov.jpg");
    cv::Mat biden_image = cv::imread("biden.jpg");

    matrix<rgb_pixel> obama_dlib_image;
    assign_image(obama_dlib_image, cv_image<bgr_pixel>(obama_image));
    matrix<rgb_pixel> biden_dlib_image;
    assign_image(biden_dlib_image, cv_image<bgr_pixel>(biden_image));

//    matrix<float, 0, 1> obama_face_encoding = face_recognition::face_encodings(obama_dlib_image, face_landmark_predictor)[0];
//    matrix<float, 0, 1> biden_face_encoding = face_recognition::face_encodings(biden_dlib_image, face_landmark_predictor)[0];

//    std::vector<matrix<float, 0, 1>> known_face_encodings = {obama_face_encoding, biden_face_encoding};
//    std::vector<string> known_face_names = {"Vladimir", "Joe Biden"};

    bool process_this_frame = true;
    cv::Mat frame;

    while (true)
    {
        video_capture.read(frame);
        if (frame.empty())
        {
            break;
        }

        if (process_this_frame)
        {
            cv_image<bgr_pixel> cimg(frame);
            matrix<rgb_pixel> img;
            assign_image(img, cimg);

            std::vector<matrix<rgb_pixel>> faces;
            std::vector<rectangle> face_locations = face_detector(img);

            for (rectangle face_location : face_locations)
            {
                auto shape = face_landmark_predictor(img, face_location);
                matrix<rgb_pixel> face_chip;
                extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                faces.push_back(move(face_chip));
            }

            /*
            std::vector<matrix<float, 0, 1>> face_encodings = face_recognition::face_encodings(faces);

            std::vector<string> face_names;
            for (auto face_encoding : face_encodings)
            {
                std::vector<bool> matches;
                for (auto known_face_encoding : known_face_encodings)
                {
                    float distance = length(known_face_encoding - face_encoding);
                    matches.push_back(distance < 0.6);  // Set an appropriate threshold for face recognition
                }

                string name = "Unknown";
                for (int i = 0; i < matches.size(); i++)
                {
                    if (matches[i])
                    {
                        name = known_face_names[i];
                        break;
                    }
                }

                face_names.push_back(name);
            }
            int i = 0;
            for (rectangle face_location : face_locations)
            {
                cv::Rect rect(face_location.left(), face_location.top(), face_location.width(), face_location.height());
                cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);

                cv::Rect name_rect(face_location.left(), face_location.bottom(), face_location.width(), 35);
                cv::rectangle(frame, name_rect, cv::Scalar(0, 0, 255), cv::FILLED);

                cv::putText(frame, face_names[i], cv::Point(face_location.left() + 6, face_location.bottom() - 6), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 255, 255), 1);

                i++;
            }
            */
        }

        process_this_frame = !process_this_frame;

        cv::imshow("Video", frame);

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    video_capture.release();
    cv::destroyAllWindows();

    return 0;
}
