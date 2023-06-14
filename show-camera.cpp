#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

int main() {
    // Загрузка предобученной модели детектора лиц
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    // Загрузка предобученной модели распознавания ключевых точек лица
    dlib::shape_predictor predictor;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

    // Запуск видеопотока с веб-камеры
    cv::VideoCapture videoCapture(0);
    if (!videoCapture.isOpened()) {
        std::cerr << "Не удалось открыть веб-камеру." << std::endl;
        return 1;
    }

    cv::Mat frame;
    cv::namedWindow("Video");

    while (true) {
        // Захват кадра с веб-камеры
        videoCapture >> frame;

        // Преобразование кадра в оттенки серого
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Конвертация кадра в dlib::matrix
        dlib::cv_image<unsigned char> dlibImage(grayFrame);

        // Обнаружение лиц на кадре
        std::vector<dlib::rectangle> faces = detector(dlibImage);

        // Обработка каждого обнаруженного лица
        for (const auto& face : faces) {
            // Распознавание ключевых точек лица
            dlib::full_object_detection landmarks = predictor(dlibImage, face);

            // Отрисовка прямоугольника вокруг лица
            cv::Rect faceRect(face.left(), face.top(), face.width(), face.height());
            cv::rectangle(frame, faceRect, cv::Scalar(0, 255, 0), 2);

            // Отрисовка ключевых точек лица
            for (unsigned int i = 0; i < landmarks.num_parts(); i++) {
                cv::Point point(landmarks.part(i).x(), landmarks.part(i).y());
                cv::circle(frame, point, 2, cv::Scalar(0, 0, 255), -1);
            }
        }

        // Отображение результата
        cv::imshow("Video", frame);

        // Выход из цикла по нажатию клавиши 'q'
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Освобождение ресурсов
    videoCapture.release();
    cv::destroyAllWindows();

    return 0;
}



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
*/

// CMD to generate executable:
// g++ webcam_opencv.cpp -o webcam_demo -I/usr/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui

// Note: check your opencv hpp files - for many users it is at /usr/local/include/opencv4
// Add more packages during compilation from the list obtained by $ pkg-config --cflags --libs opencv4





/*
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

#include "face_recognation.h"

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

    matrix<float, 0, 1> obama_face_encoding = face_recognition::face_encodings(obama_dlib_image, face_landmark_predictor)[0];
    matrix<float, 0, 1> biden_face_encoding = face_recognition::face_encodings(biden_dlib_image, face_landmark_predictor)[0];

    std::vector<matrix<float, 0, 1>> known_face_encodings = {obama_face_encoding, biden_face_encoding};
    std::vector<string> known_face_names = {"Vladimir", "Joe Biden"};

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

*/
