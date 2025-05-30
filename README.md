🔴 Hybrid Red-Eye Removal Using Multi-Feature Detection and Parallel Processing (OpenMP & CUDA)
This project implements a high-performance red-eye removal system using hybrid detection techniques (RGB, NRR, HSV) combined with parallel processing methods — including OpenMP (CPU) and CUDA (GPU). It supports benchmarking to compare sequential, CPU-parallel, and GPU-parallel execution strategies.

🚀Features
✅ Face and eye detection using Haar Cascades

Three detection methods:

RGB Thresholding

Normalized Red Ratio (NRR)

HSV Filtering

Three processing pipelines:

Sequential CPU

OpenMP (multi-core CPU)

CUDA (GPU acceleration)

✅ Benchmarks execution time for all methods

✅ Saves output images for visual and performance comparison

📁 Folder Structure

RedEyeRemoval/
├── .vscode/                 → VSCode configuration (optional)
├── include/                 → Header files
├── src/                     → C++ implementation (Sequential + OpenMP)
├── cuda/                    → CUDA kernels for red-eye correction
├── haarcascades/            → XML classifiers for face/eye detection
├── images/                  → Input image(s)
├── results/                 → Output corrected images
├── CMakeLists.txt           → Build configuration
├── main.cpp                 → Application entry point
🛠️ Prerequisites
Windows 11

Visual Studio 2022 (with Desktop C++ support)

CMake ≥ 3.18

OpenCV (tested with OpenCV 4.5+)

CUDA Toolkit (tested with 12.x)

NVIDIA GPU (Compute Capability 3.0+)

🔧 Build Instructions
Clone this repository:

bash
Copy
Edit
git clone https://github.com/yourusername/RedEyeRemoval.git
cd RedEyeRemoval
Set OpenCV_DIR and CUDA paths in your system environment if not detected automatically.

Create and build the project:

bash
Copy
Edit
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
▶️ Run the Program
bash
Copy
Edit
cd build/Release
RedEyeRemoval.exe ../../images/input1.jpg
Output images will be saved to the results/ directory.

📊 Output Files
For each input image, the program generates:

output_seq.jpg – Sequential CPU correction

output_omp.jpg – OpenMP multi-core CPU correction

output_cuda.jpg – CUDA GPU correction

📌 Benchmark Example
Method	Execution Time (ms)
Sequential	8 ms
OpenMP	5 ms
CUDA	203 ms

📚 References
OpenCV Documentation: https://docs.opencv.org/

CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

OpenMP Spec: https://www.openmp.org/

