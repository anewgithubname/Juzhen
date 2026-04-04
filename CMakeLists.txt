cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_subdirectory(external/spdlog)

# set the project directory as a macro, so that we can use it in the code
add_definitions(-DPROJECT_DIR="${CMAKE_SOURCE_DIR}")
# if you have a benchmark server running, you can specify it here
add_definitions(-DBENCHMARKSERVER="DONOTKNOW")

# do you need GPU support?
option(NVIDIA_CUDA "Build with NVIDIA CUDA backend" OFF)
option(APPLE_SILICON "Build with Apple Silicon backend" OFF)
set(HAS_CONCEPTS ON)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
 
FetchContent_Declare(ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
  GIT_TAG v6.1.9  # Replace with a version, tag, or commit hash
)
 
FetchContent_MakeAvailable(ftxui)

if(NVIDIA_CUDA)
    message("BUILD WITH CUDA SUPPORT!")
    add_definitions(-DCUDA)

    if(UNIX) 
        set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
    endif()
    
    set(CMAKE_CUDA_ARCHITECTURES 61)
    project(animated_octo_sniffle LANGUAGES CXX CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_BACKEND_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    if(NOT CUDA_BACKEND_INCLUDE_DIRS)
        set(CUDA_BACKEND_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    endif()

    set(CMAKE_CUDA_STANDARD 20)

elseif(APPLE_SILICON)
    message("BUILD WITH APPLE SILICON SUPPORT! ")
    add_definitions(-DAPPLE_SILICON)
    project(animated_octo_sniffle LANGUAGES CXX OBJCXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

    # Define shader source and output
    set(SHADER_SOURCE ${PROJECT_SOURCE_DIR}/cpp/metal/shaders.metal)
    set(SHADER_AIR ${CMAKE_BINARY_DIR}/shaders.air)
    set(SHADER_METALLIB ${CMAKE_BINARY_DIR}/default.metallib)

    # Compile Metal Shader
    add_custom_command(
        OUTPUT ${SHADER_METALLIB}
        COMMAND xcrun -sdk macosx metal -c ${SHADER_SOURCE} -o ${SHADER_AIR}
        COMMAND xcrun -sdk macosx metallib ${SHADER_AIR} -o ${SHADER_METALLIB}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling Metal shaders to ${SHADER_METALLIB}"
    )

    # Define a target to build the shaders
    add_custom_target(compile_shaders DEPENDS ${SHADER_METALLIB})

else()
    message("BUILD WITH CPU ONLY SUPPORT!")
    project(animated_octo_sniffle LANGUAGES CXX)
endif()

if(UNIX) # if we are on linux, look for any BLAS
    find_package(BLAS REQUIRED)
    message("OpenBLAS found: ${OpenBLAS_LIBRARIES}")
else() # if we are on windows, use the provided OpenBLAS library
    set(OpenBLAS_LIBRARIES "${PROJECT_SOURCE_DIR}/external/OpenBLAS/lib/libopenblas.lib")
endif()
set(OpenBLAS_INCLUDE_DIRS "external/OpenBLAS/include")

set(spdlog_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/spdlog/include")

find_package(Boost REQUIRED)

if(NVIDIA_CUDA) # add relevant backend implementations for CUDA 
    add_library(juzhen cpp/launcher.cu cpp/cumatrix.cu cpp/cukernels.cu)
elseif(APPLE_SILICON) # add relevant backend implementations for Apple Silicon
    add_library(juzhen cpp/launcher.cu cpp/metal/MPSWrapper.mm cpp/mpsmatrix.cpp)
    add_dependencies(juzhen compile_shaders)
else()
    add_library(juzhen cpp/launcher.cu)
endif()

# target_precompile_headers(juzhen PRIVATE cpp/juzhen.hpp)
target_include_directories(juzhen PUBLIC "external/spdlog/include")
target_include_directories(juzhen PUBLIC ${OpenBLAS_INCLUDE_DIRS})
target_include_directories(juzhen PUBLIC ${Boost_INCLUDE_DIRS})

if(NVIDIA_CUDA) # if we are on GPU, we need to include the cuda headers
    target_include_directories(juzhen PUBLIC ${CUDA_BACKEND_INCLUDE_DIRS})
    target_compile_options(juzhen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set(CUDA_EXTRA_LIBS ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY})

    set(CUDNN_SEARCH_INCLUDE_HINTS
        /usr/local/cuda-12.6/include
        /usr/local/cuda/include
        /usr/include)
    set(CUDNN_SEARCH_LIB_HINTS
        /usr/local/cuda-12.6/lib64
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu)

    # Support cuDNN installed via Python package nvidia-cudnn-cu12 in conda.
    if(DEFINED ENV{CONDA_PREFIX})
        file(GLOB _CONDA_CUDNN_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/lib/python*/site-packages/nvidia/cudnn/include")
        file(GLOB _CONDA_CUDNN_LIB_DIRS "$ENV{CONDA_PREFIX}/lib/python*/site-packages/nvidia/cudnn/lib")
        list(APPEND CUDNN_SEARCH_INCLUDE_HINTS ${_CONDA_CUDNN_INCLUDE_DIRS})
        list(APPEND CUDNN_SEARCH_LIB_HINTS ${_CONDA_CUDNN_LIB_DIRS})
    endif()

    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${CUDNN_SEARCH_INCLUDE_HINTS})
    find_library(CUDNN_LIBRARY
        NAMES cudnn libcudnn libcudnn.so.9
        HINTS ${CUDNN_SEARCH_LIB_HINTS})

    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
        target_include_directories(juzhen PUBLIC ${CUDNN_INCLUDE_DIR})
        add_definitions(-DCUDNN_AVAILABLE)
        list(APPEND CUDA_EXTRA_LIBS ${CUDNN_LIBRARY})
        get_filename_component(CUDNN_LIB_DIR ${CUDNN_LIBRARY} DIRECTORY)
        set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${CUDNN_LIB_DIR}")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CUDNN_LIB_DIR}")
    else()
        message(WARNING "cuDNN not found; demo_cnn_mnist_cudnn will be unavailable.")
    endif()
else()
    set_source_files_properties(cpp/launcher.cu PROPERTIES LANGUAGE CXX) # specify cuda files lanaugae to be CXX
    set_target_properties(juzhen PROPERTIES LINKER_LANGUAGE CXX)
endif()

macro(GENERATE_PROJ TARGETNAME FILE)
    # simple test executable
    add_executable(${TARGETNAME} ${FILE})
    # target_precompile_headers(${TARGETNAME} PRIVATE cpp/juzhen.hpp)
    add_dependencies(${TARGETNAME} juzhen)
    target_include_directories(${TARGETNAME} PUBLIC "external/spdlog/include")
    target_include_directories(${TARGETNAME} PUBLIC ${OpenBLAS_INCLUDE_DIRS})
    target_include_directories(${TARGETNAME} PUBLIC ${Boost_INCLUDE_DIRS})

    
    if(NVIDIA_CUDA) # if we are on GPU, we need to link the cuda libraries
        target_include_directories(${TARGETNAME} PUBLIC ${CUDA_BACKEND_INCLUDE_DIRS})
        target_link_libraries(${TARGETNAME} ftxui::screen ftxui::dom ftxui::component spdlog::spdlog ${BLAS_LIBRARIES} ${OpenBLAS_LIBRARIES} juzhen ${CUDA_EXTRA_LIBS})
        target_compile_options(${TARGETNAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    elseif(APPLE_SILICON)
        target_link_libraries(${TARGETNAME} ftxui::screen ftxui::dom ftxui::component spdlog::spdlog ${BLAS_LIBRARIES} "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders" juzhen)
        set_source_files_properties(${FILE} PROPERTIES LANGUAGE CXX)
        set_target_properties(${TARGETNAME} PROPERTIES LINKER_LANGUAGE CXX)
    else()
        target_link_libraries(${TARGETNAME} ftxui::screen ftxui::dom ftxui::component spdlog::spdlog ${BLAS_LIBRARIES} ${OpenBLAS_LIBRARIES} juzhen)
        set_source_files_properties(${FILE} PROPERTIES LANGUAGE CXX)
        set_target_properties(${TARGETNAME} PROPERTIES LINKER_LANGUAGE CXX)

        if(UNIX) # if we are on Unix, we need to specify \the language of the file to be CXX
            target_compile_options(${TARGETNAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-x c++>)
        endif()
    endif()

    if(WIN32) # if we are on Windows, we need to copy the dll to the build folder
        add_custom_command(TARGET ${TARGETNAME} POST_BUILD # Adds a post-build event to MyTest
            COMMAND ${CMAKE_COMMAND} -E copy_if_different # which executes "cmake - E copy_if_different..."
            "${PROJECT_SOURCE_DIR}/external/OpenBLAS/bin/libopenblas.dll" # <--this is in-file
            $<TARGET_FILE_DIR:${TARGETNAME}>) # <--this is out-file path
    endif()
endmacro()

GENERATE_PROJ(testbasic tests/testbasic.cu)
GENERATE_PROJ(testStackOps tests/testStackOps.cu)
GENERATE_PROJ(testEigen tests/testEigen.cu)
GENERATE_PROJ(testDataLoader tests/testDataLoader.cu)
GENERATE_PROJ(testConvLayers tests/testConvLayers.cu)
GENERATE_PROJ(testUNet tests/testUNet.cu)
GENERATE_PROJ(testUNetLearning tests/testUNetLearning.cu)
GENERATE_PROJ(testDiffusionScore tests/testDiffusionScore.cu)
GENERATE_PROJ(testMPSParity tests/testMPSParity.cu)
GENERATE_PROJ(testConvBackendParity tests/testConvBackendParity.cu)
GENERATE_PROJ(testRectifiedPathParity tests/testRectifiedPathParity.cu)
GENERATE_PROJ(testDiffusionTrajectoryCompare tests/testDiffusionTrajectoryCompare.cu)
GENERATE_PROJ(testCPUConv tests/testCPUConv.cu)
GENERATE_PROJ(testUNetScoreCPU tests/testUNetScoreCPU.cu)
GENERATE_PROJ(testDataLoaderCPU tests/testDataLoaderCPU.cu)
GENERATE_PROJ(testDiffusionTrainIterCPU tests/testDiffusionTrainIterCPU.cu)
GENERATE_PROJ(helloworld_nn examples/helloworld_nn.cu)
GENERATE_PROJ(helloworld examples/helloworld.cu)
GENERATE_PROJ(demo examples/demo.cu)
GENERATE_PROJ(knn examples/knn.cu)
GENERATE_PROJ(demo_classification examples/demo_classification.cu)
GENERATE_PROJ(demo_mnist examples/demo_mnist.cu)
GENERATE_PROJ(pagerank examples/pagerank.cu)
GENERATE_PROJ(demo_rectified examples/demo_rectified.cu)
GENERATE_PROJ(demo_rectified_infer examples/demo_rectified_infer.cu)
GENERATE_PROJ(demo_gemm examples/demo_gemm.cu)
GENERATE_PROJ(demo_cnn_mnist_cpu examples/demo_cnn_mnist_cpu.cpp)
GENERATE_PROJ(demo_cnn_mnist_mps examples/demo_cnn_mnist_mps.cu)
if(NVIDIA_CUDA)
    GENERATE_PROJ(demo_cnn_mnist_cudnn examples/demo_cnn_mnist_cudnn.cu)
endif()
GENERATE_PROJ(demo_cnn_rectified examples/demo_cnn_rectified.cu)
GENERATE_PROJ(testCNNTrainParity tests/testCNNTrainParity.cu)
# GENERATE_PROJ(wintest examples/demo_gui.cu)

enable_testing()

add_test(test0 testbasic)
add_test(test1 demo)
add_test(test2 testEigen)
add_test(test3 testDataLoader)
add_test(test4 testConvLayers)
add_test(test5 testUNet)
add_test(test6 testUNetLearning)
add_test(test7 testDiffusionScore)
add_test(test8 testMPSParity)
add_test(test9 testConvBackendParity)
add_test(test10 testRectifiedPathParity)
