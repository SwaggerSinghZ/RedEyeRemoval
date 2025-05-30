#ifndef RED_EYE_REMOVAL_HPP
#define RED_EYE_REMOVAL_HPP

#include <opencv2/opencv.hpp>
#include <string>

// Supported red-eye detection methods
enum DetectionMethod {
RGB_THRESHOLD,
NRR,
HSV
};

// Supported processing units
enum ProcessorType {
SEQUENTIAL,
OPENMP,
CUDA_PROC
};

// Core red-eye correction functions
void detectAndCorrectRedEye_CPU(const cv::Mat& input, DetectionMethod method, cv::Mat& output);
void detectAndCorrectRedEye_OpenMP(const cv::Mat& input, DetectionMethod method, cv::Mat& output);
void detectAndCorrectRedEye_CUDA(const cv::Mat& input, DetectionMethod method, cv::Mat& output);

// String name helpers
inline std::string getMethodName(DetectionMethod method) {
switch (method) {
case RGB_THRESHOLD: return "RGB";
case NRR: return "NRR";
case HSV: return "HSV";
default: return "Unknown";
}
}

inline std::string getProcessorName(ProcessorType proc) {
switch (proc) {
case SEQUENTIAL: return "seq";
case OPENMP: return "omp";
case CUDA_PROC: return "cuda";
default: return "unknown";
}
}

#endif // RED_EYE_REMOVAL_HPP