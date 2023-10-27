#include "../cpp/juzhen.hpp"

//a clean print without matrix information
void print(M matrix){

	for (int i = 0; i < matrix.num_row(); i++){
		for (int j = 0; j < matrix.num_col(); j++){
			std::cout << matrix(i, j) << " ";
		}
		std::cout << std::endl;
	}

}

int compute() {

	std::cout << "Testing C++20 Features!" << std::endl;
	std::cout << "CXX Standard: " << __cplusplus << std::endl;

	std::cout << "The following program computes a pagerank of the following web structure:"<< std::endl;
	std::cout << "1 points to 2" << std::endl;
	std::cout << "1 points to 3" << std::endl;
	std::cout << "2 points to 1" << std::endl;
	std::cout << "2 points to 3" << std::endl;
	std::cout << "3 points to 1" << std::endl;
	std::cout << std::endl;

#ifndef CPU_ONLY
	CM A(M("adjecency", { {0,1,1}, {1,0,1}, {1,0,0}} ));
	CM p(M("popularity", { { 1 / 3.0, 1 / 3.0, 1 / 3.0 } }));
	auto o13 = CM::ones(1, 3);
#else
	M A("adjecency", { {0,1,1}, {1,0,1}, {1,0,0}} );
	M p("popularity", { { 1 / 3.0, 1 / 3.0, 1 / 3.0 } });
	auto o13 = M::ones(1, 3);
#endif
	
	std::cout << "adjecency matrix:" << std::endl;
#ifndef CPU_ONLY
	print(A.to_host());
#else
	print(A);
#endif

	std::cout << std::endl;
	
	for(int i=0; i < 10; i++){
		p = p * A;
		p = p / (sum(p, 1) * o13);
	}

	std::cout << "popularity of websites:" << std::endl;
#ifndef CPU_ONLY
	print(p.to_host());
#else
	print(p);
#endif

	return 0;
}