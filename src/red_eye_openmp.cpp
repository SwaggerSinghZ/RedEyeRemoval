#include <opencv2/opencv.hpp>
#include "../include/red_eye_removal.hpp"
#include <omp.h>
#include <iostream>

void detectAndCorrectRedEye_OpenMP(const cv::Mat& input, DetectionMethod method, cv::Mat& output) {
    output = input.clone();

    // Load Haar cascade classifiers for face and eyes
    cv::CascadeClassifier faceCascade, eyeCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading face cascade\n";
        return;
    }

    if (!eyeCascade.load("haarcascade_eye.xml")) {
        std::cerr << "Error loading eye cascade\n";
        return;
    }

    // Detect faces
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(input, faces);

    for (const auto& face : faces) {
        cv::Mat faceROI = input(face);

        // Detect eyes in face region
        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(faceROI, eyes);

        for (const auto& eyeRect : eyes) {
            // Convert to absolute coordinates
            cv::Rect eyeAbs(eyeRect.x + face.x, eyeRect.y + face.y, eyeRect.width, eyeRect.height);
            cv::Mat eyeROI = output(eyeAbs);

            cv::Mat hsvROI;
            bool useHSV = false;
            if (method == HSV) {
                cv::cvtColor(eyeROI, hsvROI, cv::COLOR_BGR2HSV);
                useHSV = true;
            }

            // Parallelized red-eye detection and correction
            #pragma omp parallel for
            for (int y = 0; y < eyeROI.rows; ++y) {
                for (int x = 0; x < eyeROI.cols; ++x) {
                    cv::Vec3b& pixel = eyeROI.at<cv::Vec3b>(y, x);
                    int B = pixel[0], G = pixel[1], R = pixel[2];
                    bool isRed = false;

                    switch (method) {
                        case RGB_THRESHOLD:
                            isRed = (R > 150 && G < 100 && B < 100);
                            break;
                        case NRR:
                            isRed = (R + G + B) > 0 && (static_cast<float>(R) / (R + G + B)) > 0.6f;
                            break;
                        case HSV:
                            if (useHSV) {
                                const cv::Vec3b& hsvPixel = hsvROI.at<cv::Vec3b>(y, x);
                                int hue = hsvPixel[0], sat = hsvPixel[1], val = hsvPixel[2];
                                isRed = (hue < 10 || hue > 160) && sat > 100 && val > 100;
                            }
                            break;
                    }

                    if (isRed) {
                        int avg = (G + B) / 2;
                        pixel = cv::Vec3b(avg, avg, avg); // Grayscale fix
                    }
                }
            }
        }
    }
}
