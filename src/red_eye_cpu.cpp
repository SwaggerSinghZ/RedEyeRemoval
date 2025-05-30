#include <opencv2/opencv.hpp>
#include "../include/red_eye_removal.hpp"
#include <iostream>

void detectAndCorrectRedEye_CPU(const cv::Mat& input, DetectionMethod method, cv::Mat& output) {
    output = input.clone();

    // Load Haar cascades
    cv::CascadeClassifier faceCascade, eyeCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Failed to load face cascade.\n";
        return;
    }
    if (!eyeCascade.load("haarcascade_eye.xml")) {
        std::cerr << "Failed to load eye cascade.\n";
        return;
    }

    // Detect faces
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(input, faces);

    for (const auto& face : faces) {
        cv::Mat faceROI = input(face);
        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(faceROI, eyes);

        for (const auto& eyeRect : eyes) {
            cv::Rect eyeAbs(eyeRect.x + face.x, eyeRect.y + face.y, eyeRect.width, eyeRect.height);
            cv::Mat eyeROI = output(eyeAbs);

            // Convert to HSV only once for HSV method
            cv::Mat hsvEye;
            if (method == HSV) {
                cv::cvtColor(eyeROI, hsvEye, cv::COLOR_BGR2HSV);
            }

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
                            if ((R + G + B) > 0)
                                isRed = static_cast<float>(R) / (R + G + B) > 0.6f;
                            break;
                        case HSV: {
                            const cv::Vec3b& hsvPixel = hsvEye.at<cv::Vec3b>(y, x);
                            int hue = hsvPixel[0], sat = hsvPixel[1], val = hsvPixel[2];
                            isRed = (hue < 10 || hue > 160) && sat > 100 && val > 100;
                            break;
                        }
                    }

                    if (isRed) {
                        int avg = (G + B) / 2;
                        pixel[0] = pixel[1] = pixel[2] = static_cast<uchar>(avg);
                    }
                }
            }
        }
    }
}
