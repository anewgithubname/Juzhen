/**
 * @file layer.hpp
 * @brief layer impelementation of neural networks 
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

#ifndef LAYER2_HPP
#define LAYER2_HPP

#include <tuple>
#include <cfloat>
#include <list>  
#include "core.hpp"
#include "matrix.hpp"
#ifndef CPU_ONLY
#include "cumatrix.cuh"
#endif

namespace Juzhen
{
	/** 
	* Fully connected layer with tanh activation
	*/
	template <class D>
	class Layer {
	protected:
		Matrix<D> weights, bias, val;
		//batch size
		int nb;
		// learning rates for w and b
		double lrW, lrb;
		// do we need to update weights and bias?
		bool need_update = true;

	public:
		
		Layer(int m, int n, int nb) :
			weights("weights", m, n), bias("bias", m, 1), val("output", m, nb) {

			//intitialize parameters and gradients
			this->nb = nb;
			weights = Matrix<D>::randn(m, n) * .1;
			bias = Matrix<D>::randn(m, 1) * .1;
			val.zeros();
			lrW = .01;
			lrb = .01;
		}
		virtual const Matrix<D> grad(const Matrix<D>& input) const {
			// TODO: each time make a new matrix, not efficient
			return d_tanh(weights * input + bias * Matrix<D>::ones(1, input.num_col()));
		}

		// this function will destroy gW and gb
		void update(Matrix<D>& gW, Matrix<D>& gb) {
			weights -= lrW * std::move(gW);
			bias -= lrb * std::move(gb);
		}

		virtual void eval(const Matrix<D>& input) {
			// TODO: each time make a new matrix, not efficient
			val = tanh(weights * input + bias * Matrix<D>::ones(1, input.num_col()));
		}

		Matrix<D>& W() { return weights; };
		Matrix<D>& b() { return bias; };
		const Matrix<D>& value() { return val; }

		void print() {
			using namespace std;
			cout << weights << endl << bias << endl << lrW << endl << lrb << endl;
		}

		template <class Data>
		friend void setlr(std::list<Layer<Data>*> neuralnet, double lr);

		template <class Data> 
		friend void freeze(std::list<Layer<Data>*> neuralnet);

		template <class Data> 
		friend void unfreeze(std::list<Layer<Data>*> neuralnet);

		template <class Data>
		friend Matrix<Data> grad(std::list<Layer<Data>*> neuralnet, const Matrix<Data> &input);

		template <class Data>
		friend Matrix<Data> backprop(std::list<Layer<Data>*> neuralnet, const Matrix<Data>& input);
	};

	/**
	* Linear Layer without activation function
	*/
	template <class D>
	class LinearLayer : public Layer<D> {
	public:
		LinearLayer(int m, int n, int nb) : Layer<D>(m, n, nb) {
		}
		virtual const Matrix<D> grad(const Matrix<D>& input) const override {
			return Matrix<D>::ones(Layer<D>::weights.num_row(), input.num_col());
		}

		virtual void eval(const Matrix<D>& input) override {
			// TODO: each time make a new matrix, not efficient
			Layer<D>::val = Layer<D>::weights * input + Layer<D>::bias * Matrix<D>::ones(1, input.num_col());
		}
	};
	
	/*
	* least square layer 
	*/
	template <class D>
	class LossLayer : public Layer<D> {
		// you cannot change the output once you set it. 
		const Matrix<D>& output;
	public:
		LossLayer(int nb, const Matrix<D>& output) : Layer<D>(1, 1, nb), output(output) {
		}

		virtual const Matrix<D> grad(const Matrix<D>& input) const override {
			return 2.0* (input - output) / Layer<D>::nb;
		}

		virtual void eval(const Matrix<D>& input) override {
			Layer<D>::val = sum(square(output - input), 1) / Layer<D>::nb;
		}
	};

	/*
	* logistic layer
	*/
	template <class D>
	class LogisticLayer : public Layer<D> {
		// you cannot change the output once you set it. 
		const Matrix<D>& output;
		Matrix<D> oneK1;
	public:
		LogisticLayer(size_t nb, const Matrix<D>& output) : 
			Layer<D>(2, 2, nb), 
			output(output), 
			oneK1("oneK1", output.num_row(), 1) {
			oneK1.ones();
		}

		virtual const Matrix<D> grad(const Matrix<D>& input) const override {
			auto Z = oneK1 * sum(exp(input), 0);
			return - (output - exp(input) / std::move(Z)) / Layer<D>::nb;
		}

		virtual void eval(const Matrix<D>& input) override {
			Layer<D>::val = sum(hadmd(input, output), 0) - log(sum(exp(input),0));
			Layer<D>::val = - sum(Layer<D>::val, 1) / Layer<D>::nb;
		}
	};


	/*
	* zero-one layer
	*/
	template <class D>
	class ZeroOneLayer : public Layer<D> {
		// you cannot change the output once you set it. 
		const Matrix<D>& output;
		Matrix<D> oneK1;
	public:
		ZeroOneLayer(int nb, const Matrix<D>& output) :
			Layer<D>(1, 1, nb),
			output(output),
			oneK1("oneK1", output.num_row(), 1) {
			oneK1.ones();
		}

		virtual const Matrix<D> grad(const Matrix<D>& input) const override {
			LOG_ERROR("not supported!");
			ERROR_OUT;
		}

		virtual void eval(const Matrix<D>& input) override {
			// compute test accuracy
			double err = 0;
			int k = input.num_row();
			int nt = input.num_col();
			for (int i = 0; i < nt; i++) {
				double max = - DBL_MAX;
				int pred = -1;
				for (int j = 0; j < k; j++) {
					if (input.elem(j, i) > max) {
						max = input.elem(j, i);
						pred = j;
					}
				}
				if (output.elem(pred,i) == 0) { //TODO: dangerous
					// cout << "Prediction error: " << pred.elem(0,i) << " " << labels_t.elem(0,i) << endl;
					err++;
				}
			}
			Layer<D>::val = Matrix<D>("val", { { err / nt } });
		}
	};

	// evaluate a neural network, with input. 
	template <class D>
	const Matrix<D>& forward(std::list<Layer<D>*> neuralnet, const Matrix<D>& input) {
		if (neuralnet.size() == 0) {
			return input;
		}
		else {
			//evaluate the output and pass it on to the next layer in the stack. 
			neuralnet.back()->eval(input);
			auto& output = neuralnet.back()->value();
			neuralnet.pop_back();
			return forward(neuralnet, output);
		}
	}

	// updating the parameters in neural network.
	template <class D>
	Matrix<D> backprop(std::list<Layer<D>*> neuralnet, const Matrix<D>& input) {
		auto tt = neuralnet.back();
		neuralnet.pop_back();
		if (neuralnet.size() == 0) {
			return tt->grad(input);
		}
		else {
			// <compute curr_grad .* (W^T * pre_grad), input> 
			auto WT_prev_grad = backprop(neuralnet, tt->value());
			auto curr_grad = tt->grad(input);
			auto t = hadmd(curr_grad, std::move(WT_prev_grad));
			auto gW = t * input.T();

			// <compute curr_grad .* (W^T * pre_grad), 1> 
			auto gb = sum(t, 1);

			if(tt->need_update) {
				// gradient update
				tt->update(gW, gb);
			}

			// compute W ^ T * curr_grad
			return tt->W().T() * t;
		}
	}

	template <class D>
	Matrix<D> grad(std::list<Layer<D>*> neuralnet, const Matrix<D> &input, const Matrix<D> &W){
		/*
		* least square layer 
		*/
		class SumLayer : public Layer<D> {
			int indim = 0;
			// you cannot change the output once you set it. 
			const Matrix<D>& W;
		public:
			SumLayer(int indim, int nb, const Matrix<D> &W) : Layer<D>(2, 2, nb), W(W) {
				this->indim = indim;
			}

			virtual const Matrix<D> grad(const Matrix<D>& input) const override {
				return W;
			}

			virtual void eval(const Matrix<D>& input) override {
				Layer<D>::val = sum(sum(hadmd(W, input),1),0);
			}
		};

		auto L0 = neuralnet.back();
        freeze(neuralnet);
        
		// forward-backward pass
		forward(neuralnet, input);
        SumLayer sumL(input.num_row(), input.num_col(), W);
        neuralnet.push_front(&sumL);
        auto ret = backprop(neuralnet, input);
        neuralnet.pop_front();

        unfreeze(neuralnet);
		
		return ret;
	}

	template <class D>
	void setlr(std::list<Layer<D>*> neuralnet, double lr)
	{
		for(auto& l : neuralnet) {
			l->lrW = lr;
			l->lrb = lr;
		}
	}

	template <class D>
	void freeze(std::list<Layer<D>*> neuralnet)
	{
		for(auto& l : neuralnet) {
			l->need_update = false;
		}
	}

	template <class D>
	void unfreeze(std::list<Layer<D>*> neuralnet)
	{
		for(auto& l : neuralnet) {
			l->need_update = true;
		}
	}

}
#endif	