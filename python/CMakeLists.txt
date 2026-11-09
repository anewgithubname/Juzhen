# The extension module juzhen._core. Added by the top-level CMakeLists.txt
# when JUZHEN_PYTHON is ON (python/pyproject.toml turns it on), after the
# juzhen library and the backend options are set up.

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# NOMINSIZE: nanobind would otherwise optimize for size (/Os), which slows
# the elementwise loops of the CPU backend by about 20%.
nanobind_add_module(_core NB_STATIC NOMINSIZE src/core.cu)

# juzhen is a static library: only the backend objects the module uses are
# linked in, never launcher.cu's main(). Static libraries that go into a
# shared module need position-independent code (Linux).
set_target_properties(juzhen spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(_core PRIVATE juzhen spdlog::spdlog ${BLAS_LIBRARIES} ${OpenBLAS_LIBRARIES})

# The binding is a .cu file like the tests: compile it the same way.
if(NVIDIA_CUDA)
    target_link_libraries(_core PRIVATE ${CUDA_EXTRA_LIBS})
    target_compile_options(_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>)
elseif(ROCM_HIP)
    target_link_libraries(_core PRIVATE ${ROCM_BACKEND_LIBS})
    set_source_files_properties(src/core.cu PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x hip")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND ROCM_OFFLOAD_ARCH)
        target_compile_options(_core PRIVATE "--offload-arch=${ROCM_OFFLOAD_ARCH}")
    endif()
elseif(APPLE_SILICON)
    target_link_libraries(_core PRIVATE
        "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders")
    set_source_files_properties(src/core.cu PROPERTIES LANGUAGE CXX)
else()
    set_source_files_properties(src/core.cu PROPERTIES LANGUAGE CXX)
    if(UNIX)
        target_compile_options(_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-x c++>)
    endif()
endif()

if(MSVC)
    target_compile_options(_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

install(TARGETS _core LIBRARY DESTINATION juzhen)

# Windows does not look in PATH for the DLLs of an extension module, so
# juzhen/__init__.py adds the GPU runtime's folder, recorded here.
set(JUZHEN_DLL_DIRS "")
if(WIN32 AND NVIDIA_CUDA)
    list(APPEND JUZHEN_DLL_DIRS "${CUDAToolkit_BIN_DIR}")
elseif(WIN32 AND ROCM_HIP)
    if(DEFINED ENV{HIP_PATH})
        file(TO_CMAKE_PATH "$ENV{HIP_PATH}/bin" _rocm_bin)
    else()
        get_filename_component(_rocm_bin "${hip_DIR}/../../../bin" ABSOLUTE)
    endif()
    list(APPEND JUZHEN_DLL_DIRS "${_rocm_bin}")
endif()
set(_dll_dirs "")
foreach(_dir IN LISTS JUZHEN_DLL_DIRS)
    string(APPEND _dll_dirs "    \"${_dir}\",\n")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_build_config.py"
    "# Written by python/CMakeLists.txt when the module was built.\n"
    "backend_dll_dirs = [\n${_dll_dirs}]\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_config.py" DESTINATION juzhen)

# Metal: the compiled shaders go next to the module, where mpsInit() looks
# for them when the program (python) has none of its own.
if(APPLE_SILICON)
    install(FILES "${SHADER_METALLIB}" DESTINATION juzhen)
endif()

# Windows: the OpenBLAS DLL goes next to the module, where Python looks for it.
if(WIN32 AND NOT BLAS_FREE)
    install(FILES "${PROJECT_SOURCE_DIR}/external/OpenBLAS/bin/libopenblas.dll" DESTINATION juzhen)
endif()
