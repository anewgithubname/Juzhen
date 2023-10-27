#include "../cpp/juzhen.hpp"

#ifndef CPU_ONLY
#define __GPU_CPU__ __device__ __host__
#else
#define __GPU_CPU__
#endif

template <class T>
Matrix<T> comp_dist(const Matrix<T> &a, const Matrix<T> &b)
{
	// auto t1 = square(b);
	// std::cout << t1 << std::endl;
	// auto t2 = sum(square(b),1);
	// std::cout << t2 << std::endl;
	// auto t3 = sum(square(a), 1) * Matrix<T>::ones(1, b.num_row()) + Matrix<T>::ones(a.num_row(), 1) * sum(square(b), 1).T();
	// std::cout << t3 << std::endl;
	// auto t4 = 2 * a * b.T();
	// std::cout << t4 << std::endl;
	return sum(square(a), 1) * Matrix<T>::ones(1, b.num_row()) + Matrix<T>::ones(a.num_row(), 1) * sum(square(b), 1).T() - 2 * a * b.T();
}

template <class T>
Matrix<T> kernel_gau(Matrix<T> &&b, float sigma)
{
	return exp(-b / (2 * sigma * sigma));
}

// template <class T>
// float comp_med(const Matrix<T> &a)
// {
// 	TIC;
// 	const float *s = (float *)comp_dist(a, a).data();
// 	size_t n = a.num_row() * a.num_row();
// #ifndef CPU_ONLY
// 	thrust::device_vector<float> vec(s, s + n);
// 	thrust::sort(vec.begin(), vec.end());
// #else
// 	std::vector<T> vec(s, s + n);
// 	std::sort(vec.begin(), vec.end());
// #endif
// 	TOC;
// 	return sqrt(vec[n / 2]);
// }

/*
Find the index of the minimum element in an array.
a: the array.
len: the length of the array.
return: the index of the minimum element in the array.
*/
__GPU_CPU__ int find_min_index(float a[], int len)
{
	float min = 9999999999999999;
	int min_index = -1;
	for (int i = 0; i < len; i++)
	{
		if (a[i] < min)
		{
			min = a[i];
			min_index = i;
		}
	}
	return min_index;
}

/*
Find the indices of 5 minimum elements in an array.
a: the array.
len: the length of the array.
return: an array of 5 integers containing the indices of the 5 minimum elements in a.
*/
__GPU_CPU__ void minimumk(float a[], int len, float indices[], int k)
{
	for (int i = 0; i < k; i++)
	{
		int idx = find_min_index(a, len);
		// printf("%d %.5f\n", idx, a[idx]);
		a[idx] = 9999999999999999;
		indices[i] = (float)idx;
	}
}

template <class D>
Matrix<D> topk(const Matrix<D> &M, int k)
{
	return reduce([=] __GPU_CPU__(float *v, float *vdes, int lenv, int lendes)
				  { minimumk(v, lenv, vdes, k); },
				  M, 0, k);
}

template <class D>
Matrix<D> predict(const Matrix<D> &Idx, const Matrix<D> &L)
{
	const float *data = (float *)L.data();
	return reduce([=] __GPU_CPU__(float *v, float *vdes, int lenv, int lendes)
				  {
		int labels[100];
		// minimumk(v, lenv, vdes, k);
		for (int i = 0; i < lenv; i++)
		{
			labels[i] = (int) data[(int)v[i]];
		}
		//count the number of each label
		int count[10] = {0};
		for (int i = 0; i < lenv; i++)
		{
			count[labels[i]]++;
		}
		//find the label with the most number
		int max = 0;
		float max_index = 0;
		for (int i = 0; i < 10; i++)
		{
			if (count[i] > max)
			{
				max = count[i];
				max_index = i;
			}
		}
		vdes[0] = max_index; },
				  Idx, 0, 1);
}

int compute()
{
	std::cout << "K-Nearest Neighbour Prediction for MNIST Dataset: " << std::endl;
	global_rand_gen.seed(0);

	std::string base = PROJECT_DIR;
	std::cout << "Reading data..." << std::endl;
	Profiler *p1 = new Profiler("data loading");
#ifndef CPU_ONLY
	auto X = (CM) read<float>(base + "/X.matrix");
	auto Y = (CM) read<float>(base + "/Y_float.matrix");
	auto XT = (CM) read<float>(base + "/T.matrix");
#else
	auto X = read<float>(base + "/X.matrix");
	auto Y = read<float>(base + "/Y_float.matrix");
	auto XT = read<float>(base + "/T.matrix");
#endif

	auto YT = read<float>(base + "/YT_float.matrix");

	std::cout << "X: " << X.num_row() << "x" << X.num_col() << std::endl;
	std::cout << "Y: " << Y.num_row() << "x" << Y.num_col() << std::endl;
	std::cout << "XT: " << XT.num_row() << "x" << XT.num_col() << std::endl;
	std::cout << "YT: " << YT.num_row() << "x" << YT.num_col() << std::endl;
	delete p1;
	std::cout << "Data loaded." << std::endl
			  << std::endl;

	Profiler p("k-nearest neighbour");
	auto D = comp_dist(X.T(), XT.T());
	auto nn5 = topk(D, 7);
	auto pred = predict(nn5, Y);

#ifndef CPU_ONLY
	M hpred = pred.to_host();
#else
	M &hpred = pred;
#endif

	float miss = 0;
	for (int i = 0; i < pred.num_col(); i++)
	{
		if ((unsigned)hpred.elem(0, i) != (unsigned)YT.elem(0, i))
		{
			miss++;
		}
	}
	std::cout << "----------------------------------------" << std::endl;
	std::cout << "Misclassification Rate: " << miss / XT.num_col() << std::endl;

	return 0;
}
