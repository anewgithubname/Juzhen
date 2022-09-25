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
	nvcc -I cpp/ -O3  examples/logisticregression_MNIST_GPU.cpp cpp/cudam.cpp cpp/cukernels.cu -o bin/logi-gpu.out \
	 		 									   -lcublas -lcurand -llapack -lopenblas
logi-cpu: 
	g++ -I cpp/ -D CPU_ONLY -O3 examples/logisticregression_MNIST.cpp -o bin/logi-cpu.out -llapack -lopenblas
logi-bin: 
	g++ -I cpp/ -D CPU_ONLY -O3 examples/logisticregression_simple.cpp -o bin/logi-bin.out  -llapack -lopenblas
testing: 
	g++ -I cpp/ -D CPU_ONLY -O3 examples/test.cpp -o bin/test.out -llapack -lopenblas
helloworld:
	g++ -I cpp/ -D CPU_ONLY -O3 examples/helloworld.cpp -o bin/helloworld.out -llapack -lopenblas
helloworld-gpu:
	nvcc -I cpp/ -O3 examples/helloworld_gpu.cpp  cpp/cudam.cpp cpp/cukernels.cu -o bin/helloworld-gpu.out \
	 		 								   -lcublas -lcurand -llapack -lopenblas
helloworld-nn:
	clang++ -I cpp/ -D CPU_ONLY -O3 examples/helloworld_nn.cpp  -llapack -lopenblas -o bin/helloworld-nn.out
helloworld-nn-gpu:
	nvcc -I cpp/ -O3 examples/helloworld_nn.cpp  cpp/cudam.cpp cpp/cukernels.cu -o bin/helloworld-nn-gpu.out \
	 		 								   -lcublas -lcurand -llapack -lopenblas
else
logi-cpu: 
	# Compile without MKL
	clang++ -I cpp/ -D CPU_ONLY -std=c++20 -Ofast examples/logisticregression_MNIST.cpp -o bin/logi-cpu.out \
			-I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/ \
			-framework Accelerate
logi-bin: 
	# Compile without MKL
	clang++ -I cpp/ -D CPU_ONLY -std=c++20 -Ofast examples/logisticregression_simple.cpp -o bin/logi-bin.out \
			-I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/ \
			-framework Accelerate
testing: 
	# Compile without MKL
	clang++ -I cpp/ -D CPU_ONLY -std=c++20 -Ofast examples/test.cpp -o bin/test.out \
			-I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/ \
			-framework Accelerate
helloworld:
	clang++ -I cpp/ -D CPU_ONLY -std=c++20 -Ofast examples/helloworld.cpp -o bin/helloworld.out \
			-I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/ \
			-framework Accelerate
endif
