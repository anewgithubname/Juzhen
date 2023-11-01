cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_subdirectory(external/spdlog)

# set the project directory as a macro, so that we can use it in the code
add_definitions(-DPROJECT_DIR="${CMAKE_SOURCE_DIR}")

# do you need GPU support? CUDA 11.8 is required
set(CPU_ONLY OFF)

# are you on ARM? if so, you need to compile the external libraries yourself
set(ARM OFF)

set(CMAKE_CXX_STANDARD 17)

if(CPU_ONLY)
    message("BUILD WITHOUT GPU SUPPORT!")
    add_definitions(-DCPU_ONLY)
    project(animated_octo_sniffle LANGUAGES CXX)
else()
    message("BUILD WITH GPU SUPPORT! GOOD FOR YOU!")
    # if we are on Unix, we need to specify the CUDA toolkit location
    # CUDA 11.8 is required
    if(UNIX) 
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
        set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
        set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)
    else()
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES 61)
    project(animated_octo_sniffle LANGUAGES CXX CUDA)
    find_package(CUDA REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)

    if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
        set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
        set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
    endif()

    set(CMAKE_CUDA_FLAGS "-std=c++17")
endif()

if(UNIX) # if we are on linux, look for any BLAS
    set(BLAS_VENDOR "OpenBLAS")
    # set(BLAS_ROOT "/opt/OpenBLAS")
    find_package(BLAS REQUIRED)
else() # if we are on windows, use the provided OpenBLAS library
    set(OpenBLAS_LIBRARIES "${PROJECT_SOURCE_DIR}/external/OpenBLAS/lib/libopenblas.lib")
endif()
set(OpenBLAS_INCLUDE_DIRS "external/OpenBLAS/include")

set(spdlog_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/spdlog/include")

find_package(Boost REQUIRED)

if(NOT CPU_ONLY)
    add_library(juzhen cpp/launcher.cu cpp/cumatrix.cu cpp/cukernels.cu cpp/cumatrix.cuh cpp/juzhen.hpp cpp/core.hpp cpp/matrix.hpp cpp/helper.hpp ml/layer.hpp)
else()
    add_library(juzhen cpp/launcher.cu cpp/juzhen.hpp cpp/core.hpp cpp/matrix.hpp cpp/helper.hpp ml/layer.hpp)
endif()

target_precompile_headers(juzhen PRIVATE cpp/juzhen.hpp)
target_include_directories(juzhen PUBLIC "external/spdlog/include")
target_include_directories(juzhen PUBLIC ${OpenBLAS_INCLUDE_DIRS})
target_include_directories(juzhen PUBLIC ${Boost_INCLUDE_DIRS})

if(NOT CPU_ONLY) # if we are on GPU, we need to include the cuda headers
    target_include_directories(juzhen PUBLIC ${CUDA_INCLUDE_DIRS})
    target_compile_options(juzhen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
else()
    set_source_files_properties(cpp/launcher.cu PROPERTIES LANGUAGE CXX) # specify cuda files lanaugae to be CXX
    set_target_properties(juzhen PROPERTIES LINKER_LANGUAGE CXX)

    if(UNIX)
        target_compile_options(juzhen PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-x c++>)
    endif()
endif()

macro(GENERATE_PROJ TARGETNAME FILE)
    # simple test executable
    add_executable(${TARGETNAME} ${FILE})
    target_precompile_headers(${TARGETNAME} PRIVATE cpp/juzhen.hpp)
    add_dependencies(${TARGETNAME} juzhen)
    target_include_directories(${TARGETNAME} PUBLIC "external/spdlog/include")
    target_include_directories(${TARGETNAME} PUBLIC ${OpenBLAS_INCLUDE_DIRS})
    target_include_directories(${TARGETNAME} PUBLIC ${Boost_INCLUDE_DIRS})

    
    if(NOT CPU_ONLY) # if we are on GPU, we need to link the cuda libraries
        target_include_directories(${TARGETNAME} PUBLIC ${CUDA_INCLUDE_DIRS})
        target_link_libraries(${TARGETNAME} spdlog::spdlog ${BLAS_LIBRARIES} ${OpenBLAS_LIBRARIES} juzhen ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY})
        target_compile_options(${TARGETNAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    else()
        target_link_libraries(${TARGETNAME} spdlog::spdlog ${BLAS_LIBRARIES} ${OpenBLAS_LIBRARIES} juzhen)
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
GENERATE_PROJ(testEigen tests/testEigen.cu)
GENERATE_PROJ(helloworld_nn examples/helloworld_nn.cu)
GENERATE_PROJ(helloworld examples/helloworld.cu)
GENERATE_PROJ(demo examples/demo.cu)
GENERATE_PROJ(knn examples/knn.cu)
GENERATE_PROJ(demo_classification examples/demo_classification.cu)
GENERATE_PROJ(demo_mnist examples/demo_mnist.cu)
GENERATE_PROJ(pagerank examples/pagerank.cu)

enable_testing()

add_test(test0 testbasic)
add_test(test1 demo)
add_test(test2 testEigen)
