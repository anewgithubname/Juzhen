cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_subdirectory(external/spdlog)

# set the project directory as a macro, so that we can use it in the code
add_definitions(-DPROJECT_DIR="${CMAKE_SOURCE_DIR}")
# if you have a benchmark server running, you can specify it here
add_definitions(-DBENCHMARKSERVER="DONOTKNOW")

# do you need GPU support? CUDA 11.8 is required
set(CUDA ON)
set(APPLE_SILICON OFF)

set(HAS_CONCEPTS ON)

set(CMAKE_CXX_STANDARD 20)

if(CUDA)
    message("BUILD WITH CUDA SUPPORT!")
    add_definitions(-DCUDA)
    # if we are on Unix, we need to specify the CUDA toolkit location
    # CUDA 11.8 is required
    if(UNIX) 
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.4")
        set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)
        set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)
    else()
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES 61)
    project(animated_octo_sniffle LANGUAGES CXX CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 20)

    if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
        set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
        set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
    endif()

    set(CMAKE_CUDA_FLAGS "-std=c++20")
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

if(CUDA)
    add_library(juzhen cpp/launcher.cu cpp/cumatrix.cu cpp/cukernels.cu)
elseif(APPLE_SILICON)
    add_library(juzhen cpp/launcher.cu cpp/metal/MPSWrapper.mm cpp/mpsmatrix.cpp)
    add_dependencies(juzhen compile_shaders)
else()
    add_library(juzhen cpp/launcher.cu)
endif()

# target_precompile_headers(juzhen PRIVATE cpp/juzhen.hpp)
target_include_directories(juzhen PUBLIC "external/spdlog/include")
target_include_directories(juzhen PUBLIC ${OpenBLAS_INCLUDE_DIRS})
target_include_directories(juzhen PUBLIC ${Boost_INCLUDE_DIRS})

if(CUDA) # if we are on GPU, we need to include the cuda headers
    target_include_directories(juzhen PUBLIC ${CUDA_INCLUDE_DIRS})
    target_compile_options(juzhen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
else()
    set_source_files_properties(cpp/launcher.cu PROPERTIES LANGUAGE CXX) # specify cuda files lanaugae to be CXX
    set_target_properties(juzhen PROPERTIES LINKER_LANGUAGE CXX)

    # if(UNIX)
    #     target_compile_options(juzhen PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-x c++>)
    # endif()
endif()

macro(GENERATE_PROJ TARGETNAME FILE)
    # simple test executable
    add_executable(${TARGETNAME} ${FILE})
    # target_precompile_headers(${TARGETNAME} PRIVATE cpp/juzhen.hpp)
    add_dependencies(${TARGETNAME} juzhen)
    target_include_directories(${TARGETNAME} PUBLIC "external/spdlog/include")
    target_include_directories(${TARGETNAME} PUBLIC ${OpenBLAS_INCLUDE_DIRS})
    target_include_directories(${TARGETNAME} PUBLIC ${Boost_INCLUDE_DIRS})

    
    if(CUDA) # if we are on GPU, we need to link the cuda libraries
        target_include_directories(${TARGETNAME} PUBLIC ${CUDA_INCLUDE_DIRS})
        target_link_libraries(${TARGETNAME} spdlog::spdlog ${BLAS_LIBRARIES} ${OpenBLAS_LIBRARIES} juzhen ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY})
        target_compile_options(${TARGETNAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    elseif(APPLE_SILICON)
        target_link_libraries(${TARGETNAME} spdlog::spdlog ${BLAS_LIBRARIES} "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders" juzhen)
        set_source_files_properties(${FILE} PROPERTIES LANGUAGE CXX)
        set_target_properties(${TARGETNAME} PROPERTIES LINKER_LANGUAGE CXX)
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
GENERATE_PROJ(testDataLoader tests/testDataLoader.cu)
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
# GENERATE_PROJ(wintest examples/demo_gui.cu)

enable_testing()

add_test(test0 testbasic)
add_test(test1 demo)
add_test(test2 testEigen)
add_test(test3 testDataLoader)
