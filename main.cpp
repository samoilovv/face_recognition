#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <filesystem>

using namespace dlib;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// Путь к файлу с моделью для распознавания лиц
const std::string faceModelPath = "dlib_face_recognition_resnet_model_v1.dat";
// Путь к файлу с моделью для ориентации лиц
const std::string landmarksModelPath = "shape_predictor_5_face_landmarks.dat";
// Путь к папке с эталонами лиц
const std::string faceSamplesPath = "face_samples/";

// Отбросить расширение файла
std::string removeFileExtension(const std::string& filename) {
    size_t lastDotPos = filename.find_last_of(".");
    if (lastDotPos == std::string::npos) {
        // Если нет символа точки, возвращаем исходное имя файла
        return filename;
    } else {
        // Обрезаем имя файла до символа точки
        return filename.substr(0, lastDotPos);
    }
}

// Загрузка эталонов лиц и их меток
void loadFaceSamples(const std::string& samplesPath, std::vector<matrix<float,0,1>>& faceSamplesDescriptor, std::vector<std::string>& faceLabels,
                     dlib::frontal_face_detector& faceDetector, anet_type& faceRecognizer, const shape_predictor& sp) {
    // Получение списка файлов изображений в папке
    std::vector<std::string> fileList;

    for (const auto& entry : std::filesystem::directory_iterator(samplesPath)) {
        if (entry.is_regular_file()) {
            fileList.push_back(entry.path().string());
        }
    }

    // Загрузка каждого изображения и его метки
    std::vector<matrix<rgb_pixel>> faces;
    for (const auto& file : fileList) {
        cv::Mat image = cv::imread(file);
        if (image.empty()) {
            std::cerr << "Не удалось загрузить изображение: " << file << std::endl;
            continue;
        }

        dlib::matrix<dlib::rgb_pixel> dlibImage;
        dlib::assign_image(dlibImage, dlib::cv_image<dlib::bgr_pixel>(image));

        // Обнаружение лица на эталоне
        std::vector<dlib::rectangle> faceRects = faceDetector(dlibImage);

        if (!faceRects.empty()) {

            auto shape = sp(dlibImage, faceRects[0]);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(dlibImage, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(std::move(face_chip));
            faceLabels.push_back(removeFileExtension(file.substr(file.find_last_of('/') + 1)));
        }
    }
    faceSamplesDescriptor = faceRecognizer(faces);
}

int main() {
    // Загрузка предобученной модели детектора лиц
    dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

    //Загрузка предобученной модели ориентации лица
    shape_predictor sp;
    deserialize(landmarksModelPath) >> sp;

    // Загрузка предобученной модели для распознавания лиц
    anet_type faceRecognizer;
    dlib::deserialize(faceModelPath) >> faceRecognizer;

    // Загрузка эталонов лиц и их меток
    std::vector<matrix<float,0,1>> faceSamples;
    std::vector<std::string> faceLabels;
    loadFaceSamples(faceSamplesPath, faceSamples, faceLabels, faceDetector, faceRecognizer, sp);

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
        std::vector<dlib::rectangle> faceRects = faceDetector(dlibImage);

        // Обработка каждого обнаруженного лица
        std::vector<matrix<rgb_pixel>> faces;
        for (const auto& faceRect : faceRects) {
            // Распознавание лица
            auto shape = sp(dlibImage, faceRect);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(dlibImage, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(std::move(face_chip));
            std::vector<matrix<float,0,1>> faceDescriptor = faceRecognizer(faces);

            // Поиск наиболее близкого эталона лица
            double minDistance = std::numeric_limits<double>::max();
            double distance = 1;
            std::string closestFaceLabel;

            for (size_t i = 0; i < faceSamples.size(); ++i) {
                distance = dlib::length(faceSamples[i] - faceDescriptor[0]);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestFaceLabel = faceLabels[i];
                }
            }

            // Пороговое значение для сравнения расстояния
            double threshold = 0.6;
            std::stringstream ss;
            ss << "Distance: " << std::fixed << std::setprecision(2) << distance;
            std::string dist = ss.str();
            cv::putText(frame, dist, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255));

            // Сравнение расстояния с порогом и отрисовка прямоугольника вокруг лица и метки с идентификацией
            if (distance < threshold) {
                cv::rectangle(frame, cv::Rect(faceRect.left(), faceRect.top(), faceRect.width(), faceRect.height()), cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, closestFaceLabel, cv::Point(faceRect.left(), faceRect.top() - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
            } else {
                cv::rectangle(frame, cv::Rect(faceRect.left(), faceRect.top(), faceRect.width(), faceRect.height()), cv::Scalar(0, 0, 255), 2);
                cv::putText(frame, "Unknown Face", cv::Point(faceRect.left(), faceRect.top() - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
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


