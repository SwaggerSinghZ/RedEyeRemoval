#include <opencv2/opencv.hpp>
#include "../include/red_eye_removal.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void correctRedEyeKernel(unsigned char* data, int width, int height, int step, DetectionMethod method) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    int idx = y * step + x * 3;
    unsigned char B = data[idx];
    unsigned char G = data[idx + 1];
    unsigned char R = data[idx + 2];

    bool isRed = false;

    if (method == RGB_THRESHOLD) {
        isRed = (R > 150 && G < 100 && B < 100);
    } else if (method == NRR) {
        int sum = R + G + B;
        if (sum > 0)
            isRed = (float(R) / sum) > 0.6f;
    }

    if (isRed) {
        unsigned char avg = (G + B) / 2;
        data[idx] = data[idx + 1] = data[idx + 2] = avg;
    }
}

void detectAndCorrectRedEye_CUDA(const cv::Mat& input, DetectionMethod method, cv::Mat& output) {
    output = input.clone();

    cv::CascadeClassifier faceCascade, eyeCascade;
    faceCascade.load("haarcascade_frontalface_default.xml");
    eyeCascade.load("haarcascade_eye.xml");

    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(input, faces);

    for (const auto& face : faces) {
        cv::Mat faceROI = input(face);
        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(faceROI, eyes);

        for (const auto& eyeRect : eyes) {
            cv::Rect eyeAbs(eyeRect.x + face.x, eyeRect.y + face.y, eyeRect.width, eyeRect.height);
            cv::Mat eyeROI = output(eyeAbs);

            unsigned char* d_data;
            size_t step;
            cudaMallocPitch(&d_data, &step, eyeROI.cols * 3, eyeROI.rows);
            cudaMemcpy2D(d_data, step, eyeROI.data, eyeROI.step, eyeROI.cols * 3, eyeROI.rows, cudaMemcpyHostToDevice);

            dim3 blockSize(16, 16);
            dim3 gridSize((eyeROI.cols + 15) / 16, (eyeROI.rows + 15) / 16);
            correctRedEyeKernel<<<gridSize, blockSize>>>(d_data, eyeROI.cols, eyeROI.rows, step, method);
            cudaDeviceSynchronize();

            cudaMemcpy2D(eyeROI.data, eyeROI.step, d_data, step, eyeROI.cols * 3, eyeROI.rows, cudaMemcpyDeviceToHost);
            cudaFree(d_data);
        }
    }
}
