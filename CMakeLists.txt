cmake_minimum_required(VERSION 3.18)
project(RedEyeRemoval LANGUAGES C CXX CUDA)

# C++/CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# OpenCV
set(OpenCV_DIR "C:/opencv/build")  # Adjust if needed
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# OpenMP (CPU parallelization)
find_package(OpenMP REQUIRED COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found. OpenMP-based code may not compile.")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/red_eye_cpu.cpp
    src/red_eye_openmp.cpp
    src/red_eye_cuda.cu
)

# Add executable
add_executable(RedEyeRemoval ${SOURCES})

# CUDA compile options
set_target_properties(RedEyeRemoval PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# Link libraries
target_link_libraries(RedEyeRemoval
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

# Optional: Print summary
message(STATUS "Project successfully configured.")
message(STATUS "Sources: ${SOURCES}")
