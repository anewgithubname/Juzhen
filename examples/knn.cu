/**
 * @file knn.cu
 * @brief k nearest neighbour example
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This file contains all essential matrix operations.
 * Whatever you do, please keep it as simple as possible.
 *
    Copyright (C) 2022 Song Liu (song.liu@bristol.ac.uk)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

 */

#include "../cpp/juzhen.hpp"

#ifndef CPU_ONLY
#define __GPU_CPU__ __device__ __host__
#else
#define __GPU_CPU__
#endif

template <class T>
Matrix<T> comp_dist(const Matrix<T> &a, const Matrix<T> &b)
{
	return sum(square(a), 1) * Matrix<T>::ones(1, b.num_row()) + Matrix<T>::ones(a.num_row(), 1) * sum(square(b), 1).T() - 2 * a * b.T();
}

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

	std::string base = PROJECT_DIR + std::string("/datasets/MNIST");
	std::cout << "Reading data..." << std::endl;
	Profiler *p1 = new Profiler("data loading");

	auto Yint = read<int>(base + "/train_y.matrix");
	//convert to float for GPU computation, as GPU cannot handle int
	Matrix<float> Yhost("Y", Yint.num_row(), Yint.num_col());
	for (int i = 0; i < Yint.num_row(); i++)
	{
		for (int j = 0; j < Yint.num_col(); j++)
		{
			Yhost(i, j) = (float)Yint(i, j);
		}
	}
	auto YT = read<int>(base + "/test_y.matrix");

#ifndef CPU_ONLY
	auto X = (CM) read<float>(base + "/train_x.matrix");
	auto Y = (CM) Yhost;
	auto XT = (CM) read<float>(base + "/test_x.matrix");
#else
	auto X = read<float>(base + "/train_x.matrix");
	auto Y = std::move(Yhost);
	auto XT = read<float>(base + "/test_x.matrix");
#endif

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
