#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include "../include/red_eye_removal.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;
namespace fs = std::filesystem;

// Helper function to get top-level project directory
fs::path getProjectRoot() {
    fs::path execPath = fs::current_path();
    while (!execPath.empty()) {
        if (fs::exists(execPath / "output") && fs::exists(execPath / "include") && fs::exists(execPath / "src")) {
            return execPath;
        }
        execPath = execPath.parent_path();
    }
    return fs::current_path();  // fallback
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_image_path>" << endl;
        return 1;
    }

    string inputImagePath = argv[1];
    Mat image = imread(inputImagePath);

    if (image.empty()) {
        cerr << "Failed to load image: " << inputImagePath << endl;
        return 1;
    }

    fs::path rootDir = getProjectRoot();
    fs::path outputBase = rootDir / "output";

    // Create output subfolders
    vector<pair<string, ProcessorType>> processorFolders = {
        {"CPU", SEQUENTIAL},
        {"OpenMP", OPENMP},
        {"CUDA", CUDA_PROC}
    };

    for (const auto& [folderName, _] : processorFolders) {
        fs::create_directories(outputBase / folderName);
    }

    // Detection methods
    DetectionMethod methods[] = {RGB_THRESHOLD, NRR, HSV};

    for (const auto& [procFolder, proc] : processorFolders) {
        for (DetectionMethod method : methods) {
            if (proc == CUDA_PROC && method == HSV) {
                cout << "Skipping CUDA + HSV (unsupported in current CUDA implementation)\n";
                continue;
            }

            Mat output;
            auto start = high_resolution_clock::now();

            // Call processing function
            switch (proc) {
                case SEQUENTIAL:
                    detectAndCorrectRedEye_CPU(image, method, output);
                    break;
                case OPENMP:
                    detectAndCorrectRedEye_OpenMP(image, method, output);
                    break;
                case CUDA_PROC:
                    detectAndCorrectRedEye_CUDA(image, method, output);
                    break;
            }

            auto end = high_resolution_clock::now();
            double duration = duration_cast<milliseconds>(end - start).count();

            string procName = getProcessorName(proc);
            string methodName = getMethodName(method);
            fs::path outPath = outputBase / procFolder / ("output_" + procName + "_" + methodName + ".jpg");
            imwrite(outPath.string(), output);

            cout << "Processed (" << procName << ", " << methodName
                 << ") in " << duration << " ms -> Saved: " << outPath << endl;
        }
    }

    return 0;
}
