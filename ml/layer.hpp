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

#include <cfloat>
#include "../cpp/core.hpp" 
#include "../cpp/matrix.hpp"
#include "./util.cuh"

#ifndef CPU_ONLY
#include "../cpp/cumatrix.cuh"
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
        size_t nb;
		// learning rates for w and b
		double lrW, lrb;
		// do we need to update weights and bias?
		bool need_update = true;

		adam_state<D> adamW, adamb;

	public:
		
		Layer(size_t m, size_t n, size_t nb) :
			weights("weights", m, n), bias("bias", m, 1), val("output", m, nb), adamW(.0001, m, n), adamb(.0001, m, 1), nb(nb) {

			//intitialize parameters and gradients
			weights = Matrix<D>::randn(m, n) * .001;
			bias = Matrix<D>::randn(m, 1) * .001;
			val.zeros();
			lrW = .01;
			lrb = .01;
		}
		virtual Matrix<D> grad(const Matrix<D>& input) const {
			// TODO: each time make a new matrix, not efficient
			return d_tanh(weights * input + bias * Matrix<D>::ones(1, input.num_col()));
		}

		// this function will destroy gW and gb
		void update(Matrix<D>&& gW, Matrix<D>&& gb) {
			// adam update
			weights -= adam_update(std::move(gW), adamW);
			bias -= adam_update(std::move(gb), adamb);
			
			// //sgd update
			// weights -= lrW * std::move(gW);
			// bias -= lrb * std::move(gb);
		}

		virtual void eval(const Matrix<D>& input) {
			// TODO: each time make a new matrix, not efficient
			val = tanh(weights * input + bias * Matrix<D>::ones(1, input.num_col()));
		}

		Matrix<D>& W() { return weights; };
		Matrix<D>& b() { return bias; };
		const Matrix<D>& value() { return val; }
		adam_state<D>& adamWstate() { return adamW; }
		adam_state<D>& adambstate() { return adamb; }

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
		friend Matrix<Data> backprop(std::list<Layer<Data>*> neuralnet, const Matrix<Data>& input);
	};

	/**
	* Linear Layer without activation function
	*/
	template <class D>
	class LinearLayer : public Layer<D> {
	public:
		LinearLayer(size_t m, size_t n, size_t nb) : Layer<D>(m, n, nb) {
		}
		Matrix<D> grad(const Matrix<D>& input) const override {
			return Matrix<D>::ones(Layer<D>::weights.num_row(), input.num_col());
		}

		void eval(const Matrix<D>& input) override {
			// TODO: each time make a new matrix, not efficient
			Layer<D>::val = Layer<D>::weights * input + Layer<D>::bias * Matrix<D>::ones(1, input.num_col());
		}
	};

    /*
     * relu layer
     */
    template <class D>
    class ReluLayer : public Layer<D> {

    public:
        ReluLayer(size_t m, size_t n, size_t nb) : Layer<D>(m, n, nb) {
        }
        Matrix<D> grad(const Matrix<D>& input) const override {
            return d_relu(Layer<D>::weights * input + Layer<D>::bias * Matrix<D>::ones(1, input.num_col()));
        }

        void eval(const Matrix<D>& input) override {
            Layer<D>::val = relu(Layer<D>::weights * input + Layer<D>::bias * Matrix<D>::ones(1, input.num_col()));
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

		Matrix<D> grad(const Matrix<D>& input) const override {
			return 2.0* (input - output) / Layer<D>::nb;
		}

		void eval(const Matrix<D>& input) override {
			// square -> sum over features -> sum over batch 
			Layer<D>::val = sum(sum(square(output - input), 0), 1) / Layer<D>::nb;
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

		Matrix<D> grad(const Matrix<D>& input) const override {
			auto Z = oneK1 * sum(exp(input), 0);
			return - (output - exp(input) / std::move(Z)) / Layer<D>::nb;
		}

		void eval(const Matrix<D>& input) override {
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
		ZeroOneLayer(size_t nb, const Matrix<D>& output) :
			Layer<D>(1, 1, nb),
			output(output),
			oneK1("oneK1", output.num_row(), 1) {
			oneK1.ones();
		}

		Matrix<D> grad(const Matrix<D>&) const override {
			LOG_ERROR("you cannot differentiate zero-one layer!");
			ERROR_OUT;
		}

		void eval(const Matrix<D>& input) override {
			// compute test accuracy
			auto pred = predict_one_hot(input);
			Layer<D>::val = 1 - sum(sum(hadmd(pred, output), 0),1)/input.num_col();
		}
	};

	// evaluate a neural network, with input. 
	template <class D>
	const Matrix<D>& forward(std::list<Layer<D>*> neuralnet, const Matrix<D>& input) {
		if (neuralnet.empty()) {
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
		if (neuralnet.empty()) {
			return tt->grad(input);
		}
		else {
			// compute <curr_grad .* (W^T * pre_grad), input> 
			auto WT_prev_grad = backprop(neuralnet, tt->value());
			auto curr_grad = tt->grad(input);
			auto t = hadmd(curr_grad, std::move(WT_prev_grad));
			auto gW = t * input.T();

			// compute <curr_grad .* (W^T * pre_grad), 1> 
			auto gb = sum(t, 1);

			if(tt->need_update) {
				// gradient update
				tt->update(std::move(gW), std::move(gb));
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
			// you cannot change the output once you set it.
			const Matrix<D>& W;
		public:
            explicit SumLayer(const Matrix<D> &W): Layer<D>(2, 2, 2), W(W) {
			}

			Matrix<D> grad(const Matrix<D>&) const override {
				return W;
			}

			void eval(const Matrix<D>& input) override {
				Layer<D>::val = sum(sum(hadmd(W, input),1),0);
			}
		};

        freeze(neuralnet);
        
		// forward-backward pass
		forward(neuralnet, input);
        SumLayer sumL(W);
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

	template <class D>
	void dumpweights(std::list<Layer<D>*> neuralnet, std::string filename)
	{
		FILE *fp = fopen(filename.c_str(), "wb");

		for(auto& l : neuralnet) {
			auto layertype = typeid(*l).name();
			fwrite(layertype, sizeof(char), strlen(layertype), fp);
			write(fp, l->W());
			write(fp, l->b());
			
			// now dump the adam state
			fwrite(&l->adamWstate().iteration, sizeof(int), 1, fp);
			fwrite(&l->adamWstate().alpha, sizeof(float), 1, fp);
			fwrite(&l->adamWstate().beta1, sizeof(float), 1, fp);
			fwrite(&l->adamWstate().beta2, sizeof(float), 1, fp);
			fwrite(&l->adamWstate().eps, sizeof(float), 1, fp);

			write(fp, l->adamWstate().m);
			write(fp, l->adamWstate().v);


			fwrite(&l->adambstate().iteration, sizeof(int), 1, fp);
			fwrite(&l->adambstate().alpha, sizeof(float), 1, fp);
			fwrite(&l->adambstate().beta1, sizeof(float), 1, fp);
			fwrite(&l->adambstate().beta2, sizeof(float), 1, fp);
			fwrite(&l->adambstate().eps, sizeof(float), 1, fp);

			write(fp, l->adambstate().m);
			write(fp, l->adambstate().v);
		}
		
		fclose(fp);
	}

	template <class D>
	void loadweights(std::list<Layer<D>*> neuralnet, std::string filename){
		FILE *fp = fopen(filename.c_str(), "rb");

		for(auto& l : neuralnet) {
			auto layertype = typeid(*l).name();
			char buf[100];
			fread(buf, sizeof(char), strlen(layertype), fp); buf[strlen(layertype)] = '\0';
			if(strcmp(buf, layertype) != 0) {
				LOG_ERROR("layer type mismatch");
				ERROR_OUT;
			}
			read(fp, l->W());
			read(fp, l->b());

			std::cout << "before dump adam state" << std::endl;
			l->adamWstate().print_stats();
			// now load the adam state
			fread(&l->adamWstate().iteration, sizeof(int), 1, fp);
			fread(&l->adamWstate().alpha, sizeof(float), 1, fp);
			fread(&l->adamWstate().beta1, sizeof(float), 1, fp);
			fread(&l->adamWstate().beta2, sizeof(float), 1, fp);
			fread(&l->adamWstate().eps, sizeof(float), 1, fp);

			read(fp, l->adamWstate().m);
			read(fp, l->adamWstate().v);

			fread(&l->adambstate().iteration, sizeof(int), 1, fp);
			fread(&l->adambstate().alpha, sizeof(float), 1, fp);
			fread(&l->adambstate().beta1, sizeof(float), 1, fp);
			fread(&l->adambstate().beta2, sizeof(float), 1, fp);
			fread(&l->adambstate().eps, sizeof(float), 1, fp);

			read(fp, l->adambstate().m);
			read(fp, l->adambstate().v);
			std::cout << "after dump adam state" << std::endl;
			l->adamWstate().print_stats();

		}
		
		fclose(fp);
	}

	template <class T>
	std::vector<Matrix<T>> euler_integration(const Matrix<T>& Z0, std::list<Layer<T>*>& trainnn, int steps) {
		// start euler integration
		//std::cout << "start euler integration: " << std::endl;
		std::string base = PROJECT_DIR;
		//Profiler p("int");
		
		std::vector<Matrix<T>> ret;

		auto Zt = Z0;
		int n = Z0.num_col();

		float dt = 1.0f / steps;
		
		for (int i = 0; i < steps; i++){
			if(i % 10 == 0) {
				ret.push_back(Zt);
			}
			
			float t = (float)i/steps;    
#ifdef CPU_ONLY
			auto inpt = vstack<float>({Zt, Matrix<T>::ones(1, n)*t});
#else
			auto inpt = vstack({Zt, Matrix<T>::ones(1, n)*t});
#endif
			
			Zt += forward(trainnn, inpt) * dt;
		}

		ret.push_back(Zt);
		return ret;
	}

}

#endif
