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

#ifndef LAYER_HPP
#define LAYER_HPP
#include <cfloat>
#include <list>  
#include "juzhen.hpp"

namespace Juzhen
{
	/** 
	* Fully connected layer with tanh activation
	*/
	template <class M>
	class Layer {
	protected:
		M weights, bias, val;
		//batch size
		int nb;
		float lrW, lrb;
	public:
		Layer(int m, int n, int nb) :
			weights("weights", m, n), bias("bias", m, 1), val("output", m, nb) {

			//intitialize parameters and gradients
			this->nb = nb;
			weights = M::randn(m, n) * .1f;
			bias = M::randn(m, 1) * .1f;
			val.zeros();
			lrW = .01f;
			lrb = .01f;
		}
		virtual const M grad(const M& input) const {
			// TODO: each time make a new matrix, not efficient
			return d_tanh(weights * input + bias * M::ones(1, input.num_col()));
		}

		// this function will destroy gW and gb
		void update(M& gW, M& gb) {
			weights -= lrW * std::move(gW);
			bias -= lrb * std::move(gb);
		}

		virtual void eval(const M& input) {
			// TODO: each time make a new matrix, not efficient
			val = tanh(weights * input + bias * M::ones(1, input.num_col()));
		}

		const M& W() { return weights; };
		const M& b() { return bias; };
		const M& value() { return val; }

		void print() {
			using namespace std;
			cout << weights << endl << bias << endl;
		}
	};

	/**
	* Linear Layer without activation function
	*/
	template <class M>
	class LinearLayer : public Layer<M> {
		M o;
	public:
		LinearLayer(int m, int n, int nb) : Layer<M>(m, n, nb), o("ones", m, nb) {
			o.ones();
		}
		virtual const M grad(const M& input) const override {
			return 1.0f*o; 
		}

		virtual void eval(const M& input) override {
			// TODO: each time make a new matrix, not efficient
			Layer<M>::val = Layer<M>::weights * input + Layer<M>::bias * M::ones(1, input.num_col());
		}
	};
	
	/*
	* least square layer 
	*/
	template <class M>
	class LossLayer : public Layer<M> {
		// you cannot change the output once you set it. 
		const M& output;
	public:
		LossLayer(int nb, const M& output) : Layer<M>(1, 1, nb), output(output) {
		}

		virtual const M grad(const M& input) const override {
			return 2.0f * (input - output) / Layer<M>::nb;
		}

		virtual void eval(const M& input) override {
			Layer<M>::val = sum(square(output - input), 1) / Layer<M>::nb;
		}
	};

	/*
	* logistic layer
	*/
	template <class M>
	class LogisticLayer : public Layer<M> {
		// you cannot change the output once you set it. 
		const M& output;
		M oneK1;
	public:
		LogisticLayer(int nb, const M& output) : 
			Layer<M>(2, 2, nb), 
			output(output), 
			oneK1("oneK1", output.num_row(), 1) {
			oneK1.ones();
		}

		virtual const M grad(const M& input) const override {
			auto Z = oneK1 * sum(exp(input), 0);
			return - (output - exp(input) / std::move(Z)) / Layer<M>::nb;
		}

		virtual void eval(const M& input) override {
			Layer<M>::val = sum(hadmd(input, output), 0) - log(sum(exp(input),0));
			Layer<M>::val = - sum(Layer<M>::val, 1) / Layer<M>::nb;
		}
	};


	/*
	* zero-one layer
	*/
	template <class M>
	class ZeroOneLayer : public Layer<M> {
		// you cannot change the output once you set it. 
		const M& output;
		M oneK1;
	public:
		ZeroOneLayer(int nb, const M& output) :
			Layer<M>(1, 1, nb),
			output(output),
			oneK1("oneK1", output.num_row(), 1) {
			oneK1.ones();
		}

		virtual const M grad(const M& input) const override {
			LOG_ERROR("not supported!");
			ERROR_OUT;
		}

		virtual void eval(const M& input) override {
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
			Layer<M>::val = M("val", { { err / nt } });
		}
	};


	// evaluate a neural network, with input. 
	template<class M>
	const M& forward(std::list<Layer<M>*> neuralnet, const M& input) {
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
	template<class M>
	M backprop(std::list<Layer<M>*> neuralnet, const M& input) {
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

			// gradient update
			tt->update(gW, gb);

			// compute W ^ T * curr_grad
			return tt->W().T() * t;
		}
	}
}
#endif	