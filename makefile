# There are three build modes:
#  logi-cpu: build logistic regression using CPU
#  logi-gpu: build logistic regression using GPU
#  test: build test program

# Find the OS platform using the uname command.
Linux := $(findstring Linux, $(shell uname -s))
MacOS := $(findstring Darwin, $(shell uname -s))
Windows := $(findstring NT, $(shell uname -s))

ifdef Linux
logi-gpu:
	nvcc -I cpp/ -O3 -c cpp/helper.cpp
	nvcc -I cpp/ -O3 -I /usr/local/cuda/include -c cpp/cudam.cpp
	nvcc -I cpp/ -O3 -c cpp/cukernels.cu 
	nvcc -I cpp/ -O3  example/logisticregression_MNIST_GPU.cpp -o bin/logi-gpu.out cudam.o cukernels.o helper.o \
	 		 									   -lcublas -lcurand -llapack -lopenblas
logi-cpu: 
	g++ -I cpp/ -D CPU_ONLY -O3 -c cpp/helper.cpp
	g++ -I cpp/ -D CPU_ONLY -O3 example/logisticregression_MNIST.cpp -o bin/logi.out helper.o  -llapack -lopenblas
testing: 
	g++ -I cpp/ -D CPU_ONLY -O3 -c cpp/helper.cpp
	g++ -I cpp/ -D CPU_ONLY -O3 example/test.cpp -o bin/test.out helper.o  -llapack -lopenblas
helloworld:
	g++ -I cpp/ -D CPU_ONLY -O3 -c cpp/helper.cpp
	g++ -I cpp/ -D CPU_ONLY -O3 example/helloworld.cpp -o bin/helloworld.out helper.o  -llapack -lopenblas
helloworld-gpu:
	nvcc -I cpp/ -O3 -c cpp/helper.cpp
	nvcc -I cpp/ -O3 -c cpp/cudam.cpp
	nvcc -I cpp/ -O3 -c cpp/cukernels.cu 
	nvcc -I cpp/ -O3 example/helloworld_gpu.cpp -o bin/helloworld-gpu.out cudam.o cukernels.o helper.o \
	 		 								   -lcublas -lcurand -llapack -lopenblas
else
logi-cpu: 
	# Compile without MKL
	clang++ -I cpp/ -D CPU_ONLY -Rpass-analysis=loop-vectorize -Ofast -std=c++20 -c cpp/helper.cpp
	clang++ -I cpp/ -D CPU_ONLY -Rpass-analysis=loop-vectorize -std=c++20 -Ofast example/logisticregression_MNIST.cpp -o bin/logi.out helper.o \
			-I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/ \
			-framework Accelerate
	# # Compile using MKL
	# icc -I cpp/ -D INTEL_MKL -D CPU_ONLY -qmkl -O3 -march=coffeelake -std=c++20 -c cpp/helper.cpp
	# icc -I cpp/ -D INTEL_MKL -D CPU_ONLY -qmkl -std=c++20 -O3 -march=coffeelake example/logisticregression_MNIST.cpp -o bin/logi.out helper.o \
	# 		-I /opt/intel/oneapi/mkl/2021.4.0/include \
	# 		-L/opt/intel/oneapi/mkl/2021.4.0/lib
testing: 
	# Compile without MKL
	clang++ -I cpp/ -D CPU_ONLY -Rpass-analysis=loop-vectorize -Ofast -std=c++20 -c cpp/helper.cpp
	clang++ -I cpp/ -D CPU_ONLY -Rpass-analysis=loop-vectorize -std=c++20 -Ofast example/test.cpp -o bin/test.out helper.o \
			-I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/ \
			-framework Accelerate
	# # Compile using MKL
	# icc -I cpp/ -D INTEL_MKL -D CPU_ONLY -qmkl -std=c++20 -O3 -march=coffeelake -std=c++20 -c cpp/helper.cpp 
	# icc -I cpp/ -D INTEL_MKL -D CPU_ONLY -qmkl -std=c++20 -O3 -march=coffeelake example/test.cpp -o bin/test.out helper.o 
helloworld:
	clang++ -I cpp/ -D CPU_ONLY -Rpass-analysis=loop-vectorize -Ofast -std=c++20 -c cpp/helper.cpp
	clang++ -I cpp/ -D CPU_ONLY -Rpass-analysis=loop-vectorize -std=c++20 -Ofast example/helloworld.cpp -o bin/helloworld.out helper.o \
			-I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/ \
			-framework Accelerate
endif
