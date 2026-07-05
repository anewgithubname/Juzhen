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
#include <type_traits>
#include "../cpp/core.hpp" 
#include "../cpp/matrix.hpp"
#include "./util.cuh"

#ifdef CUDA
#include "../cpp/cumatrix.cuh"
#endif

#ifdef ROCM_HIP
#include "../cpp/hipmatrix.cuh"
#endif

#if defined(CUDA) && defined(CUDNN_AVAILABLE)
#include <cudnn.h>
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

		// Full backward pass for this layer: computes weight updates and returns
		// the gradient to propagate to the previous layer.
		// Default implementation covers all fully-connected layers.
		virtual Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream_grad) {
			auto curr_grad = grad(input);
			auto t = hadmd(curr_grad, std::move(upstream_grad));
			auto gW = t * input.T();
			auto gb = sum(t, 1);
			if (need_update) {
				update(std::move(gW), std::move(gb));
			}
			return W().T() * t;
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
		Matrix<D> output;
	public:
		LossLayer(int nb, const Matrix<D>& output) : Layer<D>(1, 1, nb), output(output) { // copy from the output
		}

		LossLayer(int nb, Matrix<D>&& output) : Layer<D>(1, 1, nb), output(output) { // move the output to the output owned by the layer
		}

		Matrix<D> grad(const Matrix<D>& input) const override {
			return 2.0 * (input - output) / Layer<D>::nb;
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
		Matrix<D> output;
		Matrix<D> oneK1;
	public:
		LogisticLayer(size_t nb, const Matrix<D>& output) : 
			Layer<D>(2, 2, nb), 
			output(output), 
			oneK1("oneK1", output.num_row(), 1) {
			oneK1.ones();
		}

		LogisticLayer(size_t nb, Matrix<D>&& output) : 
			Layer<D>(2, 2, nb), 
			output(output), 
			oneK1("oneK1", output.num_row(), 1) {
			oneK1.ones();
		}

		Matrix<D> grad(const Matrix<D>& input) const override {
			// Numerically stable softmax: subtract column-wise max before exp.
			auto mx = reduce(
				[] __GPU_CPU__(float* v, float* vdes, int lenv, int) {
					float m = -1e30f;
					for (int i = 0; i < lenv; i++) m = m > v[i] ? m : v[i];
					vdes[0] = m;
				}, input, 0, 1);                       // (1, N)
			auto shifted = input - oneK1 * mx;         // (K, N)
			auto E = exp(std::move(shifted));
			auto Z = oneK1 * sum(E, 0);
			return - (output - E / std::move(Z)) / Layer<D>::nb;
		}

		void eval(const Matrix<D>& input) override {
			// Numerically stable log-sum-exp: max subtraction prevents overflow.
			auto mx = reduce(
				[] __GPU_CPU__(float* v, float* vdes, int lenv, int) {
					float m = -1e30f;
					for (int i = 0; i < lenv; i++) m = m > v[i] ? m : v[i];
					vdes[0] = m;
				}, input, 0, 1);                       // (1, N)
			auto shifted = input - oneK1 * mx;         // (K, N)
			auto lse = log(sum(exp(std::move(shifted)), 0)) + mx; // (1, N)
			Layer<D>::val = sum(hadmd(input, output), 0) - lse;
			Layer<D>::val = - sum(Layer<D>::val, 1) / Layer<D>::nb;
		}
	};


	/*
	* zero-one layer
	*/
	template <class D>
	class ZeroOneLayer : public Layer<D> {
		// you cannot change the output once you set it. 
		Matrix<D> output;
		Matrix<D> oneK1;

	public:
		ZeroOneLayer(size_t nb, const Matrix<D>& output) :
			Layer<D>(1, 1, nb),
			output(output),
			oneK1("oneK1", output.num_row(), 1) {
			oneK1.ones();
			std::cout << "copied" << std::endl;
		}
		ZeroOneLayer(size_t nb, Matrix<D>&& output) :
			Layer<D>(1, 1, nb),
			output(output),
			oneK1("oneK1", output.num_row(), 1) {
			oneK1.ones();
			// std::cout << "moved" << std::endl;
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
			auto WT_prev_grad = backprop(neuralnet, tt->value());
			return tt->backward(input, std::move(WT_prev_grad));
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
		if (!fp) {
			LOG_ERROR("dumpweights: cannot open file '{}'", filename);
			ERROR_OUT;
		}

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
		if (!fp) {
			LOG_ERROR("loadweights: cannot open file '{}'", filename);
			ERROR_OUT;
		}

		for(auto& l : neuralnet) {
			auto layertype = typeid(*l).name();
			size_t typelen = strlen(layertype);
			std::vector<char> buf(typelen + 1, '\0');
			fread(buf.data(), sizeof(char), typelen, fp);
			if(strcmp(buf.data(), layertype) != 0) {
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
#if !defined(CUDA) && !defined(APPLE_SILIICON)
			auto inpt = vstack<float>({Zt, Matrix<T>::ones(1, n)*t});
#else
			auto inpt = vstack({Zt, Matrix<T>::ones(1, n)*t});
#endif
			
			Zt += forward(trainnn, inpt) * dt;
		}

		ret.push_back(Zt);
		return ret;
	}

#if defined(CUDA) && defined(CUDNN_AVAILABLE)
	/**
	 * Convolutional layer (conv + ReLU) backed by cuDNN.
	 *
	 * Memory layout convention (matches Matrix<CUDAfloat> column-major storage):
	 *   input  — shape (C_in  * H_in  * W_in,  N)  →  NCHW row-major for cuDNN
	 *   output — shape (C_out * H_out * W_out, N)  →  NCHW row-major for cuDNN
	 *
	 * where H_out = (H_in + 2*pad - kH) / stride + 1  (same for W_out).
	 */
	class ConvLayer : public Layer<CUDAfloat> {
		using D = CUDAfloat;

		int C_in, H_in, W_in;
		int C_out, kH, kW;
		int pad_h, pad_w, stride_h, stride_w;
		int H_out, W_out;
		int batchN;

		cudnnHandle_t               cudnn;
		cudnnTensorDescriptor_t     x_desc, y_desc, b_desc;
		cudnnFilterDescriptor_t     w_desc;
		cudnnConvolutionDescriptor_t conv_desc;

		cudnnConvolutionFwdAlgo_t       fwd_algo;
		cudnnConvolutionBwdDataAlgo_t   bwd_data_algo;
		cudnnConvolutionBwdFilterAlgo_t bwd_filt_algo;

		size_t workspace_bytes = 0;
		void*  d_workspace     = nullptr;
		bool   use_relu        = true;

		// Cast a Matrix<CUDAfloat> data pointer to float* for cuDNN.
		// Safe because CUDAfloat is a Boost strong_typedef of float with
		// identical memory layout, and the underlying storage is always mutable.
		static float* ptr(const Matrix<D>& m) {
			return const_cast<float*>(reinterpret_cast<const float*>(m.data()));
		}

		void cudnn_check(cudnnStatus_t s, const char* file, int line) {
			if (s != CUDNN_STATUS_SUCCESS) {
				LOG_ERROR("cuDNN error: {} {}:{}", cudnnGetErrorString(s), file, line);
				ERROR_OUT;
			}
		}
#define CUDNN_CHECK(expr) cudnn_check((expr), __FILE__, __LINE__)

	public:
		/**
		 * @param N       Batch size
		 * @param C_in    Input channels
		 * @param H_in    Input height
		 * @param W_in    Input width
		 * @param C_out   Number of output (filter) channels
		 * @param kH      Kernel height
		 * @param kW      Kernel width
		 * @param pad     Zero-padding applied to each spatial side (default 0)
		 * @param stride  Convolution stride (default 1)
		 */
		ConvLayer(int N, int C_in, int H_in, int W_in,
		          int C_out, int kH, int kW,
		          int pad = 0, int stride = 1, bool relu = true)
			: Layer<D>(
				  C_out * ((H_in + 2*pad - kH)/stride + 1) *
				          ((W_in + 2*pad - kW)/stride + 1),
				  1, N),
			  C_in(C_in), H_in(H_in), W_in(W_in),
			  C_out(C_out), kH(kH), kW(kW),
			  pad_h(pad), pad_w(pad), stride_h(stride), stride_w(stride),
			  H_out((H_in + 2*pad - kH)/stride + 1),
			  W_out((W_in + 2*pad - kW)/stride + 1),
			  batchN(N), use_relu(relu)
		{
			// Override base-class weight/bias/adam with convolution-appropriate sizes.
			this->weights = Matrix<D>::randn(C_out * C_in * kH * kW, 1) * 0.001f;
			this->bias    = Matrix<D>::zeros(C_out, 1);
			this->val     = Matrix<D>("conv_out", C_out * H_out * W_out, N);
			this->adamW   = adam_state<D>(0.0001, C_out * C_in * kH * kW, 1);
			this->adamb   = adam_state<D>(0.0001, C_out, 1);

			// ── cuDNN initialisation ──────────────────────────────────────────
			CUDNN_CHECK(cudnnCreate(&cudnn));
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));
			CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
			CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in,  H_in,  W_in));
			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out));
			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
			CUDNN_CHECK(cudnnSetFilter4dDescriptor(
				w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, kH, kW));
			CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
				conv_desc, pad, pad, stride, stride, 1, 1,
				CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

			// Algorithm selection (v7 API works with cuDNN 7 and 8)
			int ret;
			cudnnConvolutionFwdAlgoPerf_t       fwd_p;
			cudnnConvolutionBwdDataAlgoPerf_t   bwd_d_p;
			cudnnConvolutionBwdFilterAlgoPerf_t bwd_f_p;

			CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
				cudnn, x_desc, w_desc, conv_desc, y_desc, 1, &ret, &fwd_p));
			CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
				cudnn, w_desc, y_desc, conv_desc, x_desc, 1, &ret, &bwd_d_p));
			CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
				cudnn, x_desc, y_desc, conv_desc, w_desc, 1, &ret, &bwd_f_p));

			fwd_algo      = fwd_p.algo;
			bwd_data_algo = bwd_d_p.algo;
			bwd_filt_algo = bwd_f_p.algo;

			// Workspace — use the maximum needed by any of the three passes.
			size_t ws_fwd, ws_bwd_d, ws_bwd_f;
			CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
				cudnn, x_desc, w_desc, conv_desc, y_desc, fwd_algo, &ws_fwd));
			CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
				cudnn, w_desc, y_desc, conv_desc, x_desc, bwd_data_algo, &ws_bwd_d));
			CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
				cudnn, x_desc, y_desc, conv_desc, w_desc, bwd_filt_algo, &ws_bwd_f));

			workspace_bytes = std::max({ws_fwd, ws_bwd_d, ws_bwd_f});
			if (workspace_bytes > 0)
				CudaErrorCheck(cudaMalloc(&d_workspace, workspace_bytes));
		}

		~ConvLayer() {
			cudnnDestroyTensorDescriptor(x_desc);
			cudnnDestroyTensorDescriptor(y_desc);
			cudnnDestroyTensorDescriptor(b_desc);
			cudnnDestroyFilterDescriptor(w_desc);
			cudnnDestroyConvolutionDescriptor(conv_desc);
			cudnnDestroy(cudnn);
			if (d_workspace) cudaFree(d_workspace);
		}

		// Disable copy (cuDNN handles are not copyable).
		ConvLayer(const ConvLayer&)            = delete;
		ConvLayer& operator=(const ConvLayer&) = delete;

		/**
		 * Forward pass: val = ReLU(conv(input, W) + b)
		 *
		 * The post-ReLU output stored in val is sufficient for the backward pass
		 * because d_relu(relu(x)) == d_relu(x) for all x.
		 */
		void eval(const Matrix<D>& input) override {
			const float alpha = 1.0f, beta = 0.0f, one = 1.0f;

			// Convolution: val = conv(input, W)
			CUDNN_CHECK(cudnnConvolutionForward(
				cudnn, &alpha,
				x_desc, ptr(input),
				w_desc, ptr(this->weights),
				conv_desc, fwd_algo, d_workspace, workspace_bytes,
				&beta, y_desc, ptr(this->val)));

			// Bias: val += b  (broadcast over N, H_out, W_out)
			CUDNN_CHECK(cudnnAddTensor(
				cudnn, &alpha, b_desc, ptr(this->bias),
				&one,  y_desc, ptr(this->val)));

			// Optional ReLU activation
			if (use_relu)
				this->val = relu(Matrix<D>(this->val));
		}

		// grad() is superseded by backward(); return a dummy to satisfy the interface.
		Matrix<D> grad(const Matrix<D>&) const override {
			return Matrix<D>::ones(1, 1);
		}

		/**
		 * Backward pass.
		 *
		 * @param input         Input that was fed to eval() for this layer.
		 * @param upstream_grad Gradient from the layer above (dL/d_output).
		 * @return              Gradient to propagate further back (dL/d_input).
		 */
		Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream_grad) override {
			// Chain rule through activation:
			// With ReLU:    t = upstream_grad ⊙ d_relu(val)
			// Without ReLU: t = upstream_grad  (linear — derivative is 1)
			auto t = use_relu
				? hadmd(d_relu(Matrix<D>(this->val)), std::move(upstream_grad))
				: std::move(upstream_grad);

			const float alpha = 1.0f, beta = 0.0f;

			// dX — gradient w.r.t. layer input
			Matrix<D> dx("conv_dx", C_in * H_in * W_in, batchN);
			CUDNN_CHECK(cudnnConvolutionBackwardData(
				cudnn, &alpha,
				w_desc, ptr(this->weights),
				y_desc, ptr(t),
				conv_desc, bwd_data_algo, d_workspace, workspace_bytes,
				&beta, x_desc, ptr(dx)));

			// dW — gradient w.r.t. filter weights
			Matrix<D> dW("conv_dW", C_out * C_in * kH * kW, 1);
			CUDNN_CHECK(cudnnConvolutionBackwardFilter(
				cudnn, &alpha,
				x_desc, ptr(input),
				y_desc, ptr(t),
				conv_desc, bwd_filt_algo, d_workspace, workspace_bytes,
				&beta, w_desc, ptr(dW)));

			// db — gradient w.r.t. bias
			Matrix<D> db("conv_db", C_out, 1);
			CUDNN_CHECK(cudnnConvolutionBackwardBias(
				cudnn, &alpha,
				y_desc, ptr(t),
				&beta, b_desc, ptr(db)));

			if (this->need_update)
				this->update(std::move(dW), std::move(db));

			return dx;
		}
#undef CUDNN_CHECK
	};

	/**
	 * Transposed convolutional layer (deconv + ReLU) backed by cuDNN.
	 *
	 * Memory layout convention (matches Matrix<CUDAfloat> column-major storage):
	 *   input  - shape (C_in  * H_in  * W_in,  N)  -> NCHW row-major for cuDNN
	 *   output - shape (C_out * H_out * W_out, N)  -> NCHW row-major for cuDNN
	 *
	 * where H_out = (H_in - 1) * stride - 2 * pad + kH (same for W_out).
	 */
	class convtransLayer : public Layer<CUDAfloat> {
		using D = CUDAfloat;

		int C_in, H_in, W_in;
		int C_out, kH, kW;
		int pad_h, pad_w, stride_h, stride_w;
		int H_out, W_out;
		int batchN;

		cudnnHandle_t               cudnn;
		cudnnTensorDescriptor_t     x_desc, y_desc, b_desc;
		cudnnFilterDescriptor_t     w_desc;
		cudnnConvolutionDescriptor_t conv_desc;

		cudnnConvolutionBwdDataAlgo_t   fwd_deconv_algo;
		cudnnConvolutionFwdAlgo_t       bwd_input_algo;
		cudnnConvolutionBwdFilterAlgo_t bwd_filt_algo;

		size_t workspace_bytes = 0;
		void*  d_workspace     = nullptr;
		bool   use_relu        = true;

		static float* ptr(const Matrix<D>& m) {
			return const_cast<float*>(reinterpret_cast<const float*>(m.data()));
		}

		void cudnn_check(cudnnStatus_t s, const char* file, int line) {
			if (s != CUDNN_STATUS_SUCCESS) {
				LOG_ERROR("cuDNN error: {} {}:{}", cudnnGetErrorString(s), file, line);
				ERROR_OUT;
			}
		}
#define CUDNN_CHECK(expr) cudnn_check((expr), __FILE__, __LINE__)

	public:
		/**
		 * @param N       Batch size
		 * @param C_in    Input channels
		 * @param H_in    Input height
		 * @param W_in    Input width
		 * @param C_out   Number of output channels
		 * @param kH      Kernel height
		 * @param kW      Kernel width
		 * @param pad     Zero-padding applied to each spatial side (default 0)
		 * @param stride  Stride used by the transposed convolution (default 1)
		 */
		convtransLayer(int N, int C_in, int H_in, int W_in,
		               int C_out, int kH, int kW,
		               int pad = 0, int stride = 1, bool relu = true)
			: Layer<D>(
				  C_out * (((H_in - 1) * stride - 2 * pad + kH)) *
				          (((W_in - 1) * stride - 2 * pad + kW)),
				  1, N),
			  C_in(C_in), H_in(H_in), W_in(W_in),
			  C_out(C_out), kH(kH), kW(kW),
			  pad_h(pad), pad_w(pad), stride_h(stride), stride_w(stride),
			  H_out((H_in - 1) * stride - 2 * pad + kH),
			  W_out((W_in - 1) * stride - 2 * pad + kW),
			  batchN(N), use_relu(relu)
		{
			if (H_out <= 0 || W_out <= 0) {
				LOG_ERROR("Invalid transposed-conv output shape: H_out={}, W_out={}", H_out, W_out);
				ERROR_OUT;
			}

			// cuDNN expects filter dims (K, C, R, S). For transposed-conv
			// forward via cudnnConvolutionBackwardData, K must match C_in.
			this->weights = Matrix<D>::randn(C_in * C_out * kH * kW, 1) * 0.001f;
			this->bias    = Matrix<D>::zeros(C_out, 1);
			this->val     = Matrix<D>("convtrans_out", C_out * H_out * W_out, N);
			this->adamW   = adam_state<D>(0.0001, C_in * C_out * kH * kW, 1);
			this->adamb   = adam_state<D>(0.0001, C_out, 1);

			CUDNN_CHECK(cudnnCreate(&cudnn));
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
			CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));
			CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
			CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H_in, W_in));
			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out));
			CUDNN_CHECK(cudnnSetTensor4dDescriptor(
				b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
			CUDNN_CHECK(cudnnSetFilter4dDescriptor(
				w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_in, C_out, kH, kW));
			CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
				conv_desc, pad, pad, stride, stride, 1, 1,
				CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

			int ret;
			cudnnConvolutionBwdDataAlgoPerf_t   fwd_p;
			cudnnConvolutionFwdAlgoPerf_t       bwd_in_p;
			cudnnConvolutionBwdFilterAlgoPerf_t bwd_f_p;

			CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
				cudnn, w_desc, x_desc, conv_desc, y_desc, 1, &ret, &fwd_p));
			CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
				cudnn, y_desc, w_desc, conv_desc, x_desc, 1, &ret, &bwd_in_p));
			CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
				cudnn, y_desc, x_desc, conv_desc, w_desc, 1, &ret, &bwd_f_p));

			fwd_deconv_algo = fwd_p.algo;
			bwd_input_algo  = bwd_in_p.algo;
			bwd_filt_algo   = bwd_f_p.algo;

			size_t ws_fwd, ws_bwd_in, ws_bwd_f;
			CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
				cudnn, w_desc, x_desc, conv_desc, y_desc, fwd_deconv_algo, &ws_fwd));
			CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
				cudnn, y_desc, w_desc, conv_desc, x_desc, bwd_input_algo, &ws_bwd_in));
			CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
				cudnn, y_desc, x_desc, conv_desc, w_desc, bwd_filt_algo, &ws_bwd_f));

			workspace_bytes = std::max({ws_fwd, ws_bwd_in, ws_bwd_f});
			if (workspace_bytes > 0)
				CudaErrorCheck(cudaMalloc(&d_workspace, workspace_bytes));
		}

		~convtransLayer() {
			cudnnDestroyTensorDescriptor(x_desc);
			cudnnDestroyTensorDescriptor(y_desc);
			cudnnDestroyTensorDescriptor(b_desc);
			cudnnDestroyFilterDescriptor(w_desc);
			cudnnDestroyConvolutionDescriptor(conv_desc);
			cudnnDestroy(cudnn);
			if (d_workspace) cudaFree(d_workspace);
		}

		convtransLayer(const convtransLayer&)            = delete;
		convtransLayer& operator=(const convtransLayer&) = delete;

		void eval(const Matrix<D>& input) override {
			const float alpha = 1.0f, beta = 0.0f, one = 1.0f;

			// y = ConvTranspose(x, W)
			CUDNN_CHECK(cudnnConvolutionBackwardData(
				cudnn, &alpha,
				w_desc, ptr(this->weights),
				x_desc, ptr(input),
				conv_desc, fwd_deconv_algo, d_workspace, workspace_bytes,
				&beta, y_desc, ptr(this->val)));

			// y += b (broadcast over N, H_out, W_out)
			CUDNN_CHECK(cudnnAddTensor(
				cudnn, &alpha, b_desc, ptr(this->bias),
				&one,  y_desc, ptr(this->val)));

			if (use_relu)
				this->val = relu(Matrix<D>(this->val));
		}

		Matrix<D> grad(const Matrix<D>&) const override {
			return Matrix<D>::ones(1, 1);
		}

		Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream_grad) override {
			auto t = use_relu
				? hadmd(d_relu(Matrix<D>(this->val)), std::move(upstream_grad))
				: std::move(upstream_grad);

			const float alpha = 1.0f, beta = 0.0f;

			// dL/dx of transposed-conv input.
			Matrix<D> dx("convtrans_dx", C_in * H_in * W_in, batchN);
			CUDNN_CHECK(cudnnConvolutionForward(
				cudnn, &alpha,
				y_desc, ptr(t),
				w_desc, ptr(this->weights),
				conv_desc, bwd_input_algo, d_workspace, workspace_bytes,
				&beta, x_desc, ptr(dx)));

			// dL/dW for transposed-conv weights.
			Matrix<D> dW("convtrans_dW", C_in * C_out * kH * kW, 1);
			CUDNN_CHECK(cudnnConvolutionBackwardFilter(
				cudnn, &alpha,
				y_desc, ptr(t),
				x_desc, ptr(input),
				conv_desc, bwd_filt_algo, d_workspace, workspace_bytes,
				&beta, w_desc, ptr(dW)));

			// dL/db
			Matrix<D> db("convtrans_db", C_out, 1);
			CUDNN_CHECK(cudnnConvolutionBackwardBias(
				cudnn, &alpha,
				y_desc, ptr(t),
				&beta, b_desc, ptr(db)));

			if (this->need_update)
				this->update(std::move(dW), std::move(db));

			return dx;
		}
#undef CUDNN_CHECK
	};

	using ConvTransLayer = convtransLayer;
#endif // CUDA && CUDNN_AVAILABLE

#if defined(APPLE_SILICON)
	class ConvLayer : public Layer<MPSfloat> {
		using D = MPSfloat;

		int C_in, H_in, W_in;
		int C_out, kH, kW;
		int pad_h, pad_w, stride_h, stride_w;
		int H_out, W_out;
		int batchN;
		bool use_relu = true;

		static inline size_t idx_chw(int c, int h, int w, int H, int W) {
			return (size_t)c * (size_t)H * (size_t)W + (size_t)h * (size_t)W + (size_t)w;
		}

		static inline size_t idx_w_conv(int co, int ci, int kh, int kw, int Cin, int kH, int kW) {
			return ((size_t)co * (size_t)Cin * (size_t)kH * (size_t)kW)
				 + ((size_t)ci * (size_t)kH * (size_t)kW)
				 + ((size_t)kh * (size_t)kW)
				 + (size_t)kw;
		}

		Matrix<D> make_im2col(const Matrix<D>& input) const {
			const int K = C_in * kH * kW;
			const int P = H_out * W_out;
			Matrix<D> col("conv_im2col", K, P * batchN);
			col.zeros();

			for (int n = 0; n < batchN; ++n) {
				for (int oh = 0; oh < H_out; ++oh) {
					for (int ow = 0; ow < W_out; ++ow) {
						const int patch_col = n * P + oh * W_out + ow;
						for (int ci = 0; ci < C_in; ++ci) {
							for (int kh_i = 0; kh_i < kH; ++kh_i) {
								for (int kw_i = 0; kw_i < kW; ++kw_i) {
									const int ih = oh * stride_h - pad_h + kh_i;
									const int iw = ow * stride_w - pad_w + kw_i;
									if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in) continue;
									const int row = (ci * kH + kh_i) * kW + kw_i;
									col.elem(row, patch_col) = input.elem(idx_chw(ci, ih, iw, H_in, W_in), n);
								}
							}
						}
					}
				}
			}

			return col;
		}

		Matrix<D> flatten_weights_2d(const Matrix<D>& w) const {
			const int K = C_in * kH * kW;
			Matrix<D> w2d("conv_w2d", C_out, K);
			for (int co = 0; co < C_out; ++co) {
				for (int ci = 0; ci < C_in; ++ci) {
					for (int kh_i = 0; kh_i < kH; ++kh_i) {
						for (int kw_i = 0; kw_i < kW; ++kw_i) {
							const int col = (ci * kH + kh_i) * kW + kw_i;
							w2d.elem(co, col) = w.elem(idx_w_conv(co, ci, kh_i, kw_i, C_in, kH, kW), 0);
						}
					}
				}
			}
			return w2d;
		}

		Matrix<D> flatten_upstream(const Matrix<D>& t) const {
			const int P = H_out * W_out;
			Matrix<D> t2d("conv_t2d", C_out, P * batchN);
			for (int n = 0; n < batchN; ++n) {
				for (int co = 0; co < C_out; ++co) {
					for (int oh = 0; oh < H_out; ++oh) {
						for (int ow = 0; ow < W_out; ++ow) {
							const int patch_col = n * P + oh * W_out + ow;
							t2d.elem(co, patch_col) = t.elem(idx_chw(co, oh, ow, H_out, W_out), n);
						}
					}
				}
			}
			return t2d;
		}

		Matrix<D> unpack_output_add_bias(const Matrix<D>& y2d) const {
			const int P = H_out * W_out;
			Matrix<D> out("conv_out", C_out * P, batchN);
			for (int n = 0; n < batchN; ++n) {
				for (int co = 0; co < C_out; ++co) {
					const float b = this->bias.elem(co, 0);
					for (int oh = 0; oh < H_out; ++oh) {
						for (int ow = 0; ow < W_out; ++ow) {
							const int patch_col = n * P + oh * W_out + ow;
							out.elem(idx_chw(co, oh, ow, H_out, W_out), n) = y2d.elem(co, patch_col) + b;
						}
					}
				}
			}
			return out;
		}

		Matrix<D> col2im(const Matrix<D>& dx_col) const {
			const int P = H_out * W_out;
			Matrix<D> dx("conv_dx", C_in * H_in * W_in, batchN);
			dx.zeros();

			for (int n = 0; n < batchN; ++n) {
				for (int oh = 0; oh < H_out; ++oh) {
					for (int ow = 0; ow < W_out; ++ow) {
						const int patch_col = n * P + oh * W_out + ow;
						for (int ci = 0; ci < C_in; ++ci) {
							for (int kh_i = 0; kh_i < kH; ++kh_i) {
								for (int kw_i = 0; kw_i < kW; ++kw_i) {
									const int ih = oh * stride_h - pad_h + kh_i;
									const int iw = ow * stride_w - pad_w + kw_i;
									if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in) continue;
									const int row = (ci * kH + kh_i) * kW + kw_i;
									dx.elem(idx_chw(ci, ih, iw, H_in, W_in), n) += dx_col.elem(row, patch_col);
								}
							}
						}
					}
				}
			}

			return dx;
		}

		Matrix<D> pack_weight_grad(const Matrix<D>& dW2d) const {
			Matrix<D> dW("conv_dW", C_out * C_in * kH * kW, 1);
			for (int co = 0; co < C_out; ++co) {
				for (int ci = 0; ci < C_in; ++ci) {
					for (int kh_i = 0; kh_i < kH; ++kh_i) {
						for (int kw_i = 0; kw_i < kW; ++kw_i) {
							const int col = (ci * kH + kh_i) * kW + kw_i;
							dW.elem(idx_w_conv(co, ci, kh_i, kw_i, C_in, kH, kW), 0) = dW2d.elem(co, col);
						}
					}
				}
			}
			return dW;
		}

		static float* ptr(const Matrix<D>& m) {
			return const_cast<float*>(reinterpret_cast<const float*>(m.data()));
		}

		Matrix<float> make_im2col(const Matrix<float>& input) const {
			const int K = C_in * kH * kW;
			const int P = H_out * W_out;
			Matrix<float> col("conv_im2col", K, P * batchN);
			col.zeros();

			for (int n = 0; n < batchN; ++n) {
				for (int oh = 0; oh < H_out; ++oh) {
					for (int ow = 0; ow < W_out; ++ow) {
						const int patch_col = n * P + oh * W_out + ow;
						for (int ci = 0; ci < C_in; ++ci) {
							for (int kh = 0; kh < kH; ++kh) {
								for (int kw = 0; kw < kW; ++kw) {
									const int ih = oh * stride_h - pad_h + kh;
									const int iw = ow * stride_w - pad_w + kw;
									const int row = (ci * kH + kh) * kW + kw;
									if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
										col.elem(row, patch_col) = input.elem(idx_chw(ci, ih, iw, H_in, W_in), n);
									}
								}
							}
						}
					}
				}
			}
			return col;
		}

		Matrix<float> flatten_weights_2d(const Matrix<float>& w) const {
			const int K = C_in * kH * kW;
			Matrix<float> w2d("conv_w2d", C_out, K);
			for (int co = 0; co < C_out; ++co) {
				for (int ci = 0; ci < C_in; ++ci) {
					for (int kh = 0; kh < kH; ++kh) {
						for (int kw = 0; kw < kW; ++kw) {
							const int row = co;
							const int col = (ci * kH + kh) * kW + kw;
							w2d.elem(row, col) = w.elem(idx_w_conv(co, ci, kh, kw, C_in, kH, kW), 0);
						}
					}
				}
			}
			return w2d;
		}

		Matrix<float> unflatten_output(const Matrix<float>& y2d, const Matrix<float>& b) const {
			const int P = H_out * W_out;
			Matrix<float> out("conv_out_host", C_out * P, batchN);
			for (int n = 0; n < batchN; ++n) {
				for (int co = 0; co < C_out; ++co) {
					for (int oh = 0; oh < H_out; ++oh) {
						for (int ow = 0; ow < W_out; ++ow) {
							const int patch_col = n * P + oh * W_out + ow;
							const size_t out_row = idx_chw(co, oh, ow, H_out, W_out);
							out.elem(out_row, n) = y2d.elem(co, patch_col) + b.elem(co, 0);
						}
					}
				}
			}
			return out;
		}

		Matrix<float> conv_input_grad_from_cols(const Matrix<float>& dx_col) const {
			const int P = H_out * W_out;
			Matrix<float> dx("conv_dx_host", C_in * H_in * W_in, batchN);
			dx.zeros();

			for (int n = 0; n < batchN; ++n) {
				for (int oh = 0; oh < H_out; ++oh) {
					for (int ow = 0; ow < W_out; ++ow) {
						const int patch_col = n * P + oh * W_out + ow;
						for (int ci = 0; ci < C_in; ++ci) {
							for (int kh = 0; kh < kH; ++kh) {
								for (int kw = 0; kw < kW; ++kw) {
									const int ih = oh * stride_h - pad_h + kh;
									const int iw = ow * stride_w - pad_w + kw;
									if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in) continue;
									const int row = (ci * kH + kh) * kW + kw;
									dx.elem(idx_chw(ci, ih, iw, H_in, W_in), n) += dx_col.elem(row, patch_col);
								}
							}
						}
					}
				}
			}
			return dx;
		}

		Matrix<float> flatten_upstream(const Matrix<float>& t) const {
			const int P = H_out * W_out;
			Matrix<float> t2d("conv_t2d", C_out, P * batchN);
			for (int n = 0; n < batchN; ++n) {
				for (int co = 0; co < C_out; ++co) {
					for (int oh = 0; oh < H_out; ++oh) {
						for (int ow = 0; ow < W_out; ++ow) {
							const int patch_col = n * P + oh * W_out + ow;
							t2d.elem(co, patch_col) = t.elem(idx_chw(co, oh, ow, H_out, W_out), n);
						}
					}
				}
			}
			return t2d;
		}

		Matrix<float> pack_weight_grad(const Matrix<float>& dW2d) const {
			Matrix<float> dW("conv_dW_host", C_out * C_in * kH * kW, 1);
			for (int co = 0; co < C_out; ++co) {
				for (int ci = 0; ci < C_in; ++ci) {
					for (int kh = 0; kh < kH; ++kh) {
						for (int kw = 0; kw < kW; ++kw) {
							const int col = (ci * kH + kh) * kW + kw;
							dW.elem(idx_w_conv(co, ci, kh, kw, C_in, kH, kW), 0) = dW2d.elem(co, col);
						}
					}
				}
			}
			return dW;
		}

	public:
		ConvLayer(int N, int C_in, int H_in, int W_in,
		          int C_out, int kH, int kW,
		          int pad = 0, int stride = 1, bool relu = true)
			: Layer<D>(
				  C_out * ((H_in + 2*pad - kH)/stride + 1) *
				          ((W_in + 2*pad - kW)/stride + 1),
				  1, N),
			  C_in(C_in), H_in(H_in), W_in(W_in),
			  C_out(C_out), kH(kH), kW(kW),
			  pad_h(pad), pad_w(pad), stride_h(stride), stride_w(stride),
			  H_out((H_in + 2*pad - kH)/stride + 1),
			  W_out((W_in + 2*pad - kW)/stride + 1),
			  batchN(N), use_relu(relu)
		{
			this->weights = Matrix<D>::randn(C_out * C_in * kH * kW, 1) * 0.001f;
			this->bias    = Matrix<D>::zeros(C_out, 1);
			this->val     = Matrix<D>("conv_out", C_out * H_out * W_out, N);
			this->adamW   = adam_state<D>(0.0001, C_out * C_in * kH * kW, 1);
			this->adamb   = adam_state<D>(0.0001, C_out, 1);
		}

		void eval(const Matrix<D>& input) override {
			auto wh = this->weights.to_host();
			auto w2d_h = flatten_weights_2d(wh);

			const int K = C_in * kH * kW;
			const int P = H_out * W_out;
			Matrix<D> col("conv_im2col", K, P * batchN);
			mpsIm2col(ptr(input), ptr(col), batchN, C_in, H_in, W_in,
			          kH, kW, pad_h, pad_w, stride_h, stride_w, H_out, W_out);

			Matrix<D> w2d(w2d_h);
			Matrix<D> y2d = w2d * col;
			Matrix<D> out("conv_out", C_out * P, batchN);
			mpsConv2dOutputAddBias(ptr(y2d), ptr(this->bias), ptr(out), batchN, C_out, H_out, W_out);
			this->val = std::move(out);

			if (use_relu) {
				this->val = relu(Matrix<D>(this->val));
			}
		}

		Matrix<D> grad(const Matrix<D>&) const override {
			return Matrix<D>::ones(1, 1);
		}

		Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream_grad) override {
			auto t = use_relu
				? hadmd(d_relu(Matrix<D>(this->val)), std::move(upstream_grad))
				: std::move(upstream_grad);

			auto wh = this->weights.to_host();
			auto w2d_h = flatten_weights_2d(wh);

			const int K = C_in * kH * kW;
			const int P = H_out * W_out;

			Matrix<D> col("conv_im2col", K, P * batchN);
			mpsIm2col(ptr(input), ptr(col), batchN, C_in, H_in, W_in,
			          kH, kW, pad_h, pad_w, stride_h, stride_w, H_out, W_out);

			Matrix<D> t2d("conv_t2d", C_out, P * batchN);
			mpsPackFeatureMap2D(ptr(t), ptr(t2d), batchN, C_out, P);

			Matrix<D> w2d(w2d_h);
			Matrix<D> dW2d = t2d * col.T();
			Matrix<D> dx_col = w2d.T() * t2d;
			Matrix<D> dx("conv_dx", C_in * H_in * W_in, batchN);
			mpsCol2im(ptr(dx_col), ptr(dx), batchN, C_in, H_in, W_in,
			          kH, kW, pad_h, pad_w, stride_h, stride_w, H_out, W_out);

			Matrix<D> db = sum(t2d, 1);
			auto dW2d_h = dW2d.to_host();

			if (this->need_update) {
				this->update(Matrix<D>(pack_weight_grad(dW2d_h)), std::move(db));
			}

			return dx;
		}
	};

	class convtransLayer : public Layer<MPSfloat> {
		using D = MPSfloat;

		int C_in, H_in, W_in;
		int C_out, kH, kW;
		int pad_h, pad_w, stride_h, stride_w;
		int H_out, W_out;
		int batchN;
		bool use_relu = true;

		Matrix<D> flatten_weights_2d() const {
			const int rows = C_out * kH * kW;
			Matrix<D> w2d("deconv_w2d", rows, C_in);
			for (int ci = 0; ci < C_in; ++ci) {
				mpsCopyMatrixBlock(ptr(this->weights), ptr(w2d),
					C_in * rows, 1, false,
					ci * rows, 0,
					rows, 1,
					rows, C_in, false,
					0, ci);
			}
			return w2d;
		}

		Matrix<D> pack_weight_grad(const Matrix<D>& dW2d) const {
			const int rows = C_out * kH * kW;
			Matrix<D> dW("convtrans_dW", C_in * rows, 1);
			for (int ci = 0; ci < C_in; ++ci) {
				mpsCopyMatrixBlock(ptr(dW2d), ptr(dW),
					rows, C_in, false,
					0, ci,
					rows, 1,
					C_in * rows, 1, false,
					ci * rows, 0);
			}
			return dW;
		}

		Matrix<D> unpack_2d_feature_map(const Matrix<D>& packed, int C, int H, int W) const {
			Matrix<D> out("deconv_unpacked", C * H * W, batchN);
			Matrix<D> zero_bias("deconv_zero_bias", C, 1);
			zero_bias.zeros();
			mpsConv2dOutputAddBias(ptr(packed), ptr(zero_bias), ptr(out), batchN, C, H, W);
			return out;
		}

		static float* ptr(const Matrix<D>& m) {
			return const_cast<float*>(reinterpret_cast<const float*>(m.data()));
		}

	public:
		convtransLayer(int N, int C_in, int H_in, int W_in,
		               int C_out, int kH, int kW,
		               int pad = 0, int stride = 1, bool relu = true)
			: Layer<D>(
				  C_out * (((H_in - 1) * stride - 2 * pad + kH)) *
				          (((W_in - 1) * stride - 2 * pad + kW)),
				  1, N),
			  C_in(C_in), H_in(H_in), W_in(W_in),
			  C_out(C_out), kH(kH), kW(kW),
			  pad_h(pad), pad_w(pad), stride_h(stride), stride_w(stride),
			  H_out((H_in - 1) * stride - 2 * pad + kH),
			  W_out((W_in - 1) * stride - 2 * pad + kW),
			  batchN(N), use_relu(relu)
		{
			if (H_out <= 0 || W_out <= 0) {
				LOG_ERROR("Invalid transposed-conv output shape: H_out={}, W_out={}", H_out, W_out);
				ERROR_OUT;
			}

			this->weights = Matrix<D>::randn(C_in * C_out * kH * kW, 1) * 0.001f;
			this->bias    = Matrix<D>::zeros(C_out, 1);
			this->val     = Matrix<D>("convtrans_out", C_out * H_out * W_out, N);
			this->adamW   = adam_state<D>(0.0001, C_in * C_out * kH * kW, 1);
			this->adamb   = adam_state<D>(0.0001, C_out, 1);
		}

		void eval(const Matrix<D>& input) override {
			const int P_in = H_in * W_in;
			Matrix<D> x2d("deconv_x2d", C_in, batchN * P_in);
			mpsPackFeatureMap2D(ptr(input), ptr(x2d), batchN, C_in, P_in);

			Matrix<D> w2d = flatten_weights_2d();
			Matrix<D> patches = w2d * x2d;

			Matrix<D> out_no_bias("deconv_out_nobias", C_out * H_out * W_out, batchN);
			mpsCol2im(ptr(patches), ptr(out_no_bias), batchN, C_out, H_out, W_out,
			          kH, kW, pad_h, pad_w, stride_h, stride_w, H_in, W_in);

			const int P_out = H_out * W_out;
			Matrix<D> packed_out("deconv_out_packed", C_out, batchN * P_out);
			mpsPackFeatureMap2D(ptr(out_no_bias), ptr(packed_out), batchN, C_out, P_out);

			this->val = Matrix<D>("convtrans_out", C_out * H_out * W_out, batchN);
			mpsConv2dOutputAddBias(ptr(packed_out), ptr(this->bias), ptr(this->val), batchN, C_out, H_out, W_out);
			if (use_relu) {
				this->val = relu(Matrix<D>(this->val));
			}
		}

		Matrix<D> grad(const Matrix<D>&) const override {
			return Matrix<D>::ones(1, 1);
		}

		Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream_grad) override {
			auto t = use_relu
				? hadmd(d_relu(Matrix<D>(this->val)), std::move(upstream_grad))
				: std::move(upstream_grad);

			const int P_in = H_in * W_in;
			const int P_out = H_out * W_out;

			Matrix<D> x2d("deconv_x2d", C_in, batchN * P_in);
			mpsPackFeatureMap2D(ptr(input), ptr(x2d), batchN, C_in, P_in);

			Matrix<D> w2d = flatten_weights_2d();

			Matrix<D> tp("deconv_tp", C_out * kH * kW, batchN * P_in);
			mpsIm2col(ptr(t), ptr(tp), batchN, C_out, H_out, W_out,
			         kH, kW, pad_h, pad_w, stride_h, stride_w, H_in, W_in);

			Matrix<D> dW2d = tp * x2d.T();
			Matrix<D> dx2d = w2d.T() * tp;

			Matrix<D> dx = unpack_2d_feature_map(dx2d, C_in, H_in, W_in);

			Matrix<D> t2d("deconv_t2d", C_out, batchN * P_out);
			mpsPackFeatureMap2D(ptr(t), ptr(t2d), batchN, C_out, P_out);
			Matrix<D> db = sum(t2d, 1);

			if (this->need_update) {
				this->update(pack_weight_grad(dW2d), std::move(db));
			}

			return dx;
		}
	};

	using ConvTransLayer = convtransLayer;
#endif // APPLE_SILICON

#if defined(ROCM_HIP)
	/**
	 * Convolutional layer (conv + optional ReLU) for ROCm/HIP backend.
	 * Uses im2col + rocBLAS GEMM (no MIOpen dependency).
	 *
	 * Memory layout: column-major, input (C_in*H_in*W_in, N), output (C_out*H_out*W_out, N).
	 * Weights stored as 1-D column (C_out*C_in*kH*kW, 1); in memory this is (K, C_out)
	 * column-major where K = C_in*kH*kW.
	 */
	class ConvLayer : public Layer<ROCMfloat> {
		using D = ROCMfloat;

		int C_in, H_in, W_in;
		int C_out, kH, kW;
		int pad_h, pad_w, stride_h, stride_w;
		int H_out, W_out;
		int batchN;
		bool use_relu = true;

		static float* ptr(const Matrix<D>& m) {
			return const_cast<float*>(reinterpret_cast<const float*>(m.data()));
		}

	public:
		ConvLayer(int N, int C_in, int H_in, int W_in,
		          int C_out, int kH, int kW,
		          int pad = 0, int stride = 1, bool relu = true)
			: Layer<D>(
				  C_out * ((H_in + 2*pad - kH)/stride + 1) *
				          ((W_in + 2*pad - kW)/stride + 1),
				  1, N),
			  C_in(C_in), H_in(H_in), W_in(W_in),
			  C_out(C_out), kH(kH), kW(kW),
			  pad_h(pad), pad_w(pad), stride_h(stride), stride_w(stride),
			  H_out((H_in + 2*pad - kH)/stride + 1),
			  W_out((W_in + 2*pad - kW)/stride + 1),
			  batchN(N), use_relu(relu)
		{
			if (H_out <= 0 || W_out <= 0) {
				LOG_ERROR("Invalid conv output shape: H_out={}, W_out={}", H_out, W_out);
				ERROR_OUT;
			}
			this->weights = Matrix<D>::randn(C_out * C_in * kH * kW, 1) * 0.001f;
			this->bias    = Matrix<D>::zeros(C_out, 1);
			this->val     = Matrix<D>("conv_out", C_out * H_out * W_out, N);
			this->adamW   = adam_state<D>(0.0001, C_out * C_in * kH * kW, 1);
			this->adamb   = adam_state<D>(0.0001, C_out, 1);
		}

		void eval(const Matrix<D>& input) override {
			const int K = C_in * kH * kW;
			const int P = H_out * W_out;

			// 1. im2col: input (C_in*H_in*W_in, N) → col (K, P*N)
			Matrix<D> col("conv_col", K, P * batchN);
			RocmIm2col(ptr(input), ptr(col),
			           C_in, H_in, W_in, kH, kW,
			           pad_h, pad_w, stride_h, stride_w,
			           H_out, W_out, batchN);

			// 2. GEMM: y2d(C_out, P*N) = W(K, C_out)^T * col(K, P*N)
			//    Weights in memory are (K, C_out) col-major; use transA.
			Matrix<D> y2d("conv_y2d", C_out, P * batchN);
			RocmGemm(ptr(this->weights), ptr(col), ptr(y2d),
			         C_out, P * batchN, K,
			         true, false,
			         K, K, C_out);

			// 3. Reshape + bias: y2d(C_out, P*N) → val(C_out*P, N) + bias
			RocmConvForwardReshapeBias(ptr(y2d), ptr(this->val), ptr(this->bias),
			                            C_out, P, batchN);

			// 4. Optional ReLU
			if (use_relu)
				this->val = relu(Matrix<D>(this->val));
		}

		Matrix<D> grad(const Matrix<D>&) const override {
			return Matrix<D>::ones(1, 1);
		}

		Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream_grad) override {
			auto t = use_relu
				? hadmd(d_relu(Matrix<D>(this->val)), std::move(upstream_grad))
				: std::move(upstream_grad);

			const int K = C_in * kH * kW;
			const int P = H_out * W_out;

			// 1. Reshape upstream: t(C_out*P, N) → t2d(C_out, P*N)
			Matrix<D> t2d("conv_t2d", C_out, P * batchN);
			RocmConvBackwardReshape(ptr(t), ptr(t2d), C_out, P, batchN);

			// 2. im2col on forward input
			Matrix<D> col("conv_col", K, P * batchN);
			RocmIm2col(ptr(input), ptr(col),
			           C_in, H_in, W_in, kH, kW,
			           pad_h, pad_w, stride_h, stride_w,
			           H_out, W_out, batchN);

			// 3. dW: stored(K, C_out) = col(K, P*N) * t2d(C_out, P*N)^T
			Matrix<D> dW("conv_dW", C_out * K, 1);
			RocmGemm(ptr(col), ptr(t2d), ptr(dW),
			         K, C_out, P * batchN,
			         false, true,
			         K, C_out, K);

			// 4. dx_col(K, P*N) = W_stored(K, C_out) * t2d(C_out, P*N)
			Matrix<D> dx_col("conv_dx_col", K, P * batchN);
			RocmGemm(ptr(this->weights), ptr(t2d), ptr(dx_col),
			         K, P * batchN, C_out,
			         false, false,
			         K, C_out, K);

			// 5. col2im: dx_col(K, P*N) → dx(C_in*H_in*W_in, N)
			Matrix<D> dx("conv_dx", C_in * H_in * W_in, batchN);
			RocmCol2im(ptr(dx_col), ptr(dx),
			           C_in, H_in, W_in, kH, kW,
			           pad_h, pad_w, stride_h, stride_w,
			           H_out, W_out, batchN);

			// 6. Bias gradient: db = sum(t2d, 1) → (C_out, 1)
			Matrix<D> db = sum(t2d, 1);

			if (this->need_update)
				this->update(std::move(dW), std::move(db));

			return dx;
		}
	};

	/**
	 * Transposed convolutional layer for ROCm/HIP backend.
	 * Uses GEMM + scatter/gather kernels.
	 *
	 * Weights stored as (C_in*C_out*kH*kW, 1); in memory this is
	 * (C_out*kH*kW, C_in) column-major.
	 */
	class convtransLayer : public Layer<ROCMfloat> {
		using D = ROCMfloat;

		int C_in, H_in, W_in;
		int C_out, kH, kW;
		int pad_h, pad_w, stride_h, stride_w;
		int H_out, W_out;
		int batchN;
		bool use_relu = true;

		static float* ptr(const Matrix<D>& m) {
			return const_cast<float*>(reinterpret_cast<const float*>(m.data()));
		}

	public:
		convtransLayer(int N, int C_in, int H_in, int W_in,
		               int C_out, int kH, int kW,
		               int pad = 0, int stride = 1, bool relu = true)
			: Layer<D>(
				  C_out * ((H_in - 1) * stride - 2 * pad + kH) *
				          ((W_in - 1) * stride - 2 * pad + kW),
				  1, N),
			  C_in(C_in), H_in(H_in), W_in(W_in),
			  C_out(C_out), kH(kH), kW(kW),
			  pad_h(pad), pad_w(pad), stride_h(stride), stride_w(stride),
			  H_out((H_in - 1) * stride - 2 * pad + kH),
			  W_out((W_in - 1) * stride - 2 * pad + kW),
			  batchN(N), use_relu(relu)
		{
			if (H_out <= 0 || W_out <= 0) {
				LOG_ERROR("Invalid transposed-conv output shape: H_out={}, W_out={}", H_out, W_out);
				ERROR_OUT;
			}
			this->weights = Matrix<D>::randn(C_in * C_out * kH * kW, 1) * 0.001f;
			this->bias    = Matrix<D>::zeros(C_out, 1);
			this->val     = Matrix<D>("convtrans_out", C_out * H_out * W_out, N);
			this->adamW   = adam_state<D>(0.0001, C_in * C_out * kH * kW, 1);
			this->adamb   = adam_state<D>(0.0001, C_out, 1);
		}

		void eval(const Matrix<D>& input) override {
			const int P_in = H_in * W_in;
			const int patch_rows = C_out * kH * kW;

			// 1. Reshape input: (C_in*P_in, N) → x2d(C_in, P_in*N)
			Matrix<D> x2d("deconv_x2d", C_in, P_in * batchN);
			RocmConvTransReshape(ptr(input), ptr(x2d), C_in, P_in, batchN, 0);

			// 2. GEMM: patches(patch_rows, P_in*N) = W(patch_rows, C_in) * x2d(C_in, P_in*N)
			Matrix<D> patches("deconv_patches", patch_rows, P_in * batchN);
			RocmGemm(ptr(this->weights), ptr(x2d), ptr(patches),
			         patch_rows, P_in * batchN, C_in,
			         false, false,
			         patch_rows, C_in, patch_rows);

			// 3. Scatter patches → output + bias
			RocmConvTransScatter(ptr(patches), ptr(this->val), ptr(this->bias),
			                      C_out, H_out, W_out, H_in, W_in,
			                      kH, kW, pad_h, pad_w, stride_h, stride_w, batchN);

			if (use_relu)
				this->val = relu(Matrix<D>(this->val));
		}

		Matrix<D> grad(const Matrix<D>&) const override {
			return Matrix<D>::ones(1, 1);
		}

		Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream_grad) override {
			auto t = use_relu
				? hadmd(d_relu(Matrix<D>(this->val)), std::move(upstream_grad))
				: std::move(upstream_grad);

			const int P_in = H_in * W_in;
			const int patch_rows = C_out * kH * kW;

			// 1. Gather upstream: t(C_out*H_out*W_out, N) → tp(patch_rows, P_in*N)
			Matrix<D> tp("deconv_tp", patch_rows, P_in * batchN);
			RocmConvTransGather(ptr(t), ptr(tp),
			                     C_out, H_out, W_out, H_in, W_in,
			                     kH, kW, pad_h, pad_w, stride_h, stride_w, batchN);

			// 2. Pack input: (C_in*P_in, N) → x2d(C_in, P_in*N)
			Matrix<D> x2d("deconv_x2d", C_in, P_in * batchN);
			RocmConvTransReshape(ptr(input), ptr(x2d), C_in, P_in, batchN, 0);

			// 3. dW: W_stored(patch_rows, C_in) = tp(patch_rows, P_in*N) * x2d(C_in, P_in*N)^T
			Matrix<D> dW("deconv_dW", C_in * C_out * kH * kW, 1);
			RocmGemm(ptr(tp), ptr(x2d), ptr(dW),
			         patch_rows, C_in, P_in * batchN,
			         false, true,
			         patch_rows, C_in, patch_rows);

			// 4. dx2d(C_in, P_in*N) = W(patch_rows, C_in)^T * tp(patch_rows, P_in*N)
			Matrix<D> dx2d("deconv_dx2d", C_in, P_in * batchN);
			RocmGemm(ptr(this->weights), ptr(tp), ptr(dx2d),
			         C_in, P_in * batchN, patch_rows,
			         true, false,
			         patch_rows, patch_rows, C_in);

			// 5. Unpack: dx2d(C_in, P_in*N) → dx(C_in*P_in, N)
			Matrix<D> dx("deconv_dx", C_in * P_in, batchN);
			RocmConvTransReshape(ptr(dx2d), ptr(dx), C_in, P_in, batchN, 1);

			// 6. Bias gradient: sum over spatial+batch from upstream t
			//    Reshape t → (C_out, H_out*W_out*N), then row-sum
			const int P_out = H_out * W_out;
			Matrix<D> t2d("deconv_t2d", C_out, P_out * batchN);
			RocmConvBackwardReshape(ptr(t), ptr(t2d), C_out, P_out, batchN);
			Matrix<D> db = sum(t2d, 1);

			if (this->need_update)
				this->update(std::move(dW), std::move(db));

			return dx;
		}
	};

	using ConvTransLayer = convtransLayer;
#endif // ROCM_HIP

#if !defined(CUDA) && !defined(APPLE_SILICON) && !defined(ROCM_HIP)
	class ConvLayer : public Layer<float> {
		using D = float;

		int C_in, H_in, W_in;
		int C_out, kH, kW;
		int pad_h, pad_w, stride_h, stride_w;
		int H_out, W_out;
		int batchN;
		bool use_relu = true;

		static inline size_t idx_chw(int c, int h, int w, int H, int W) {
			return (size_t)c * (size_t)H * (size_t)W + (size_t)h * (size_t)W + (size_t)w;
		}

		static inline size_t idx_w_conv(int co, int ci, int kh, int kw, int Cin, int kH, int kW) {
			return ((size_t)co * (size_t)Cin * (size_t)kH * (size_t)kW)
				 + ((size_t)ci * (size_t)kH * (size_t)kW)
				 + ((size_t)kh * (size_t)kW)
				 + (size_t)kw;
		}

		Matrix<D> make_im2col(const Matrix<D>& input) const {
			const int K = C_in * kH * kW;
			const int P = H_out * W_out;
			Matrix<D> col("conv_im2col", K, P * batchN);
			col.zeros();

			for (int n = 0; n < batchN; ++n) {
				for (int oh = 0; oh < H_out; ++oh) {
					for (int ow = 0; ow < W_out; ++ow) {
						const int patch_col = n * P + oh * W_out + ow;
						for (int ci = 0; ci < C_in; ++ci) {
							for (int kh_i = 0; kh_i < kH; ++kh_i) {
								for (int kw_i = 0; kw_i < kW; ++kw_i) {
									const int ih = oh * stride_h - pad_h + kh_i;
									const int iw = ow * stride_w - pad_w + kw_i;
									if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in) continue;
									const int row = (ci * kH + kh_i) * kW + kw_i;
									col.elem(row, patch_col) = input.elem(idx_chw(ci, ih, iw, H_in, W_in), n);
								}
							}
						}
					}
				}
			}

			return col;
		}

		Matrix<D> flatten_weights_2d(const Matrix<D>& w) const {
			const int K = C_in * kH * kW;
			Matrix<D> w2d("conv_w2d", C_out, K);
			for (int co = 0; co < C_out; ++co) {
				for (int ci = 0; ci < C_in; ++ci) {
					for (int kh_i = 0; kh_i < kH; ++kh_i) {
						for (int kw_i = 0; kw_i < kW; ++kw_i) {
							const int col = (ci * kH + kh_i) * kW + kw_i;
							w2d.elem(co, col) = w.elem(idx_w_conv(co, ci, kh_i, kw_i, C_in, kH, kW), 0);
						}
					}
				}
			}
			return w2d;
		}

		Matrix<D> flatten_upstream(const Matrix<D>& t) const {
			const int P = H_out * W_out;
			Matrix<D> t2d("conv_t2d", C_out, P * batchN);
			for (int n = 0; n < batchN; ++n) {
				for (int co = 0; co < C_out; ++co) {
					for (int oh = 0; oh < H_out; ++oh) {
						for (int ow = 0; ow < W_out; ++ow) {
							const int patch_col = n * P + oh * W_out + ow;
							t2d.elem(co, patch_col) = t.elem(idx_chw(co, oh, ow, H_out, W_out), n);
						}
					}
				}
			}
			return t2d;
		}

		Matrix<D> unpack_output_add_bias(const Matrix<D>& y2d) const {
			const int P = H_out * W_out;
			Matrix<D> out("conv_out", C_out * P, batchN);
			for (int n = 0; n < batchN; ++n) {
				for (int co = 0; co < C_out; ++co) {
					const float b = this->bias.elem(co, 0);
					for (int oh = 0; oh < H_out; ++oh) {
						for (int ow = 0; ow < W_out; ++ow) {
							const int patch_col = n * P + oh * W_out + ow;
							out.elem(idx_chw(co, oh, ow, H_out, W_out), n) = y2d.elem(co, patch_col) + b;
						}
					}
				}
			}
			return out;
		}

		Matrix<D> col2im(const Matrix<D>& dx_col) const {
			const int P = H_out * W_out;
			Matrix<D> dx("conv_dx", C_in * H_in * W_in, batchN);
			dx.zeros();

			for (int n = 0; n < batchN; ++n) {
				for (int oh = 0; oh < H_out; ++oh) {
					for (int ow = 0; ow < W_out; ++ow) {
						const int patch_col = n * P + oh * W_out + ow;
						for (int ci = 0; ci < C_in; ++ci) {
							for (int kh_i = 0; kh_i < kH; ++kh_i) {
								for (int kw_i = 0; kw_i < kW; ++kw_i) {
									const int ih = oh * stride_h - pad_h + kh_i;
									const int iw = ow * stride_w - pad_w + kw_i;
									if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in) continue;
									const int row = (ci * kH + kh_i) * kW + kw_i;
									dx.elem(idx_chw(ci, ih, iw, H_in, W_in), n) += dx_col.elem(row, patch_col);
								}
							}
						}
					}
				}
			}

			return dx;
		}

		Matrix<D> pack_weight_grad(const Matrix<D>& dW2d) const {
			Matrix<D> dW("conv_dW", C_out * C_in * kH * kW, 1);
			for (int co = 0; co < C_out; ++co) {
				for (int ci = 0; ci < C_in; ++ci) {
					for (int kh_i = 0; kh_i < kH; ++kh_i) {
						for (int kw_i = 0; kw_i < kW; ++kw_i) {
							const int col = (ci * kH + kh_i) * kW + kw_i;
							dW.elem(idx_w_conv(co, ci, kh_i, kw_i, C_in, kH, kW), 0) = dW2d.elem(co, col);
						}
					}
				}
			}
			return dW;
		}

	public:
		ConvLayer(int N, int C_in, int H_in, int W_in,
		          int C_out, int kH, int kW,
		          int pad = 0, int stride = 1, bool relu = true)
			: Layer<D>(
				  C_out * ((H_in + 2*pad - kH)/stride + 1) *
				          ((W_in + 2*pad - kW)/stride + 1),
				  1, N),
			  C_in(C_in), H_in(H_in), W_in(W_in),
			  C_out(C_out), kH(kH), kW(kW),
			  pad_h(pad), pad_w(pad), stride_h(stride), stride_w(stride),
			  H_out((H_in + 2*pad - kH)/stride + 1),
			  W_out((W_in + 2*pad - kW)/stride + 1),
			  batchN(N), use_relu(relu)
		{
			if (H_out <= 0 || W_out <= 0) {
				LOG_ERROR("Invalid conv output shape: H_out={}, W_out={}", H_out, W_out);
				ERROR_OUT;
			}

			this->weights = Matrix<D>::randn(C_out * C_in * kH * kW, 1) * 0.001f;
			this->bias    = Matrix<D>::zeros(C_out, 1);
			this->val     = Matrix<D>("conv_out", C_out * H_out * W_out, N);
			this->adamW   = adam_state<D>(0.0001, C_out * C_in * kH * kW, 1);
			this->adamb   = adam_state<D>(0.0001, C_out, 1);
		}

		void eval(const Matrix<D>& input) override {
			const Matrix<D> col = make_im2col(input);
			const Matrix<D> w2d = flatten_weights_2d(this->weights);
			const Matrix<D> y2d = w2d * col;
			this->val = unpack_output_add_bias(y2d);
			if (use_relu) {
				this->val = relu(Matrix<D>(this->val));
			}
		}

		Matrix<D> grad(const Matrix<D>&) const override {
			return Matrix<D>::ones(1, 1);
		}

		Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream_grad) override {
			auto t = use_relu
				? hadmd(d_relu(Matrix<D>(this->val)), std::move(upstream_grad))
				: std::move(upstream_grad);

			const Matrix<D> col = make_im2col(input);
			const Matrix<D> w2d = flatten_weights_2d(this->weights);
			const Matrix<D> t2d = flatten_upstream(t);

			const Matrix<D> dW2d = t2d * col.T();
			const Matrix<D> dx_col = w2d.T() * t2d;
			Matrix<D> db = sum(t2d, 1);
			Matrix<D> dx = col2im(dx_col);

			if (this->need_update) {
				this->update(pack_weight_grad(dW2d), std::move(db));
			}

			return dx;
		}
	};

	class convtransLayer : public Layer<float> {
		using D = float;

		int C_in, H_in, W_in;
		int C_out, kH, kW;
		int pad_h, pad_w, stride_h, stride_w;
		int H_out, W_out;
		int batchN;
		bool use_relu = true;

		static inline size_t idx_chw(int c, int h, int w, int H, int W) {
			return (size_t)c * (size_t)H * (size_t)W + (size_t)h * (size_t)W + (size_t)w;
		}

		static inline size_t idx_w_deconv(int ci, int co, int kh, int kw, int Cout, int kH, int kW) {
			return ((size_t)ci * (size_t)Cout * (size_t)kH * (size_t)kW)
				 + ((size_t)co * (size_t)kH * (size_t)kW)
				 + ((size_t)kh * (size_t)kW)
				 + (size_t)kw;
		}

		Matrix<D> pack_input_2d(const Matrix<D>& input) const {
			const int P_in = H_in * W_in;
			Matrix<D> x2d("deconv_x2d", C_in, batchN * P_in);
			for (int n = 0; n < batchN; ++n) {
				for (int ci = 0; ci < C_in; ++ci) {
					for (int ih = 0; ih < H_in; ++ih) {
						for (int iw = 0; iw < W_in; ++iw) {
							const int col = n * P_in + ih * W_in + iw;
							x2d.elem(ci, col) = input.elem(idx_chw(ci, ih, iw, H_in, W_in), n);
						}
					}
				}
			}
			return x2d;
		}

		Matrix<D> flatten_weights_2d(const Matrix<D>& w) const {
			Matrix<D> w2d("deconv_w2d", C_out * kH * kW, C_in);
			for (int ci = 0; ci < C_in; ++ci) {
				for (int co = 0; co < C_out; ++co) {
					for (int kh_i = 0; kh_i < kH; ++kh_i) {
						for (int kw_i = 0; kw_i < kW; ++kw_i) {
							const int row = (co * kH + kh_i) * kW + kw_i;
							w2d.elem(row, ci) = w.elem(idx_w_deconv(ci, co, kh_i, kw_i, C_out, kH, kW), 0);
						}
					}
				}
			}
			return w2d;
		}

		Matrix<D> unpack_forward_from_patches(const Matrix<D>& patches) const {
			Matrix<D> out("convtrans_out", C_out * H_out * W_out, batchN);
			out.zeros();

			for (int n = 0; n < batchN; ++n) {
				for (int co = 0; co < C_out; ++co) {
					const float b = this->bias.elem(co, 0);
					for (int oh = 0; oh < H_out; ++oh) {
						for (int ow = 0; ow < W_out; ++ow) {
							out.elem(idx_chw(co, oh, ow, H_out, W_out), n) = b;
						}
					}
				}
			}

			for (int n = 0; n < batchN; ++n) {
				for (int ih = 0; ih < H_in; ++ih) {
					for (int iw = 0; iw < W_in; ++iw) {
						const int col = n * H_in * W_in + ih * W_in + iw;
						for (int co = 0; co < C_out; ++co) {
							for (int kh_i = 0; kh_i < kH; ++kh_i) {
								for (int kw_i = 0; kw_i < kW; ++kw_i) {
									const int oh = ih * stride_h - pad_h + kh_i;
									const int ow = iw * stride_w - pad_w + kw_i;
									if (oh < 0 || oh >= H_out || ow < 0 || ow >= W_out) continue;
									const int row = (co * kH + kh_i) * kW + kw_i;
									out.elem(idx_chw(co, oh, ow, H_out, W_out), n) += patches.elem(row, col);
								}
							}
						}
					}
				}
			}

			return out;
		}

		Matrix<D> build_upstream_patches(const Matrix<D>& t) const {
			const int P_in = H_in * W_in;
			Matrix<D> tp("deconv_tp", C_out * kH * kW, batchN * P_in);
			tp.zeros();

			for (int n = 0; n < batchN; ++n) {
				for (int ih = 0; ih < H_in; ++ih) {
					for (int iw = 0; iw < W_in; ++iw) {
						const int col = n * P_in + ih * W_in + iw;
						for (int co = 0; co < C_out; ++co) {
							for (int kh_i = 0; kh_i < kH; ++kh_i) {
								for (int kw_i = 0; kw_i < kW; ++kw_i) {
									const int oh = ih * stride_h - pad_h + kh_i;
									const int ow = iw * stride_w - pad_w + kw_i;
									if (oh < 0 || oh >= H_out || ow < 0 || ow >= W_out) continue;
									const int row = (co * kH + kh_i) * kW + kw_i;
									tp.elem(row, col) = t.elem(idx_chw(co, oh, ow, H_out, W_out), n);
								}
							}
						}
					}
				}
			}

			return tp;
		}

		Matrix<D> unpack_dx_from_2d(const Matrix<D>& dx2d) const {
			const int P_in = H_in * W_in;
			Matrix<D> dx("convtrans_dx", C_in * H_in * W_in, batchN);
			for (int n = 0; n < batchN; ++n) {
				for (int ci = 0; ci < C_in; ++ci) {
					for (int ih = 0; ih < H_in; ++ih) {
						for (int iw = 0; iw < W_in; ++iw) {
							const int col = n * P_in + ih * W_in + iw;
							dx.elem(idx_chw(ci, ih, iw, H_in, W_in), n) = dx2d.elem(ci, col);
						}
					}
				}
			}
			return dx;
		}

		Matrix<D> pack_weight_grad(const Matrix<D>& dW2d) const {
			Matrix<D> dW("convtrans_dW", C_in * C_out * kH * kW, 1);
			for (int ci = 0; ci < C_in; ++ci) {
				for (int co = 0; co < C_out; ++co) {
					for (int kh_i = 0; kh_i < kH; ++kh_i) {
						for (int kw_i = 0; kw_i < kW; ++kw_i) {
							const int row = (co * kH + kh_i) * kW + kw_i;
							dW.elem(idx_w_deconv(ci, co, kh_i, kw_i, C_out, kH, kW), 0) = dW2d.elem(row, ci);
						}
					}
				}
			}
			return dW;
		}

	public:
		convtransLayer(int N, int C_in, int H_in, int W_in,
		               int C_out, int kH, int kW,
		               int pad = 0, int stride = 1, bool relu = true)
			: Layer<D>(
				  C_out * (((H_in - 1) * stride - 2 * pad + kH)) *
				          (((W_in - 1) * stride - 2 * pad + kW)),
				  1, N),
			  C_in(C_in), H_in(H_in), W_in(W_in),
			  C_out(C_out), kH(kH), kW(kW),
			  pad_h(pad), pad_w(pad), stride_h(stride), stride_w(stride),
			  H_out((H_in - 1) * stride - 2 * pad + kH),
			  W_out((W_in - 1) * stride - 2 * pad + kW),
			  batchN(N), use_relu(relu)
		{
			if (H_out <= 0 || W_out <= 0) {
				LOG_ERROR("Invalid transposed-conv output shape: H_out={}, W_out={}", H_out, W_out);
				ERROR_OUT;
			}

			this->weights = Matrix<D>::randn(C_in * C_out * kH * kW, 1) * 0.001f;
			this->bias    = Matrix<D>::zeros(C_out, 1);
			this->val     = Matrix<D>("convtrans_out", C_out * H_out * W_out, N);
			this->adamW   = adam_state<D>(0.0001, C_in * C_out * kH * kW, 1);
			this->adamb   = adam_state<D>(0.0001, C_out, 1);
		}

		void eval(const Matrix<D>& input) override {
			const Matrix<D> x2d = pack_input_2d(input);
			const Matrix<D> w2d = flatten_weights_2d(this->weights);
			const Matrix<D> patches = w2d * x2d;
			this->val = unpack_forward_from_patches(patches);
			if (use_relu) {
				this->val = relu(Matrix<D>(this->val));
			}
		}

		Matrix<D> grad(const Matrix<D>&) const override {
			return Matrix<D>::ones(1, 1);
		}

		Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream_grad) override {
			auto t = use_relu
				? hadmd(d_relu(Matrix<D>(this->val)), std::move(upstream_grad))
				: std::move(upstream_grad);

			const Matrix<D> x2d = pack_input_2d(input);
			const Matrix<D> w2d = flatten_weights_2d(this->weights);
			const Matrix<D> tp = build_upstream_patches(t);

			Matrix<D> db("convtrans_db", C_out, 1);
			db.zeros();
			for (int n = 0; n < batchN; ++n) {
				for (int co = 0; co < C_out; ++co) {
					for (int oh = 0; oh < H_out; ++oh) {
						for (int ow = 0; ow < W_out; ++ow) {
							db.elem(co, 0) += t.elem(idx_chw(co, oh, ow, H_out, W_out), n);
						}
					}
				}
			}

			const Matrix<D> dW2d = tp * x2d.T();
			const Matrix<D> dx2d = w2d.T() * tp;
			Matrix<D> dx = unpack_dx_from_2d(dx2d);

			if (this->need_update) {
				this->update(pack_weight_grad(dW2d), std::move(db));
			}

			return dx;
		}
	};

	using ConvTransLayer = convtransLayer;
#endif // !CUDA && !APPLE_SILICON && !ROCM_HIP

	// ── Transformer Layer ─────────────────────────────────────────────────

	// Column-wise softmax: softmax along dim-0 (each column independently).
	// M shape: (rows, cols).  Returns same shape.
	template <class D>
	Matrix<D> col_softmax(const Matrix<D>& M) {
		// max per column for numerical stability
		auto mx = reduce(
			[] __GPU_CPU__(float* v, float* vdes, int lenv, int) {
				float m = -1e30f;
				for (int i = 0; i < lenv; i++) m = m > v[i] ? m : v[i];
				vdes[0] = m;
			}, M, 0, 1);                           // (1, cols)
		auto shifted = M - Matrix<D>::ones(M.num_row(), 1) * mx;
		auto e = exp(std::move(shifted));
		auto s = sum(e, 0);                        // (1, cols)
		return hadmd(e, Matrix<D>::ones(M.num_row(), 1) * (elemwise(
			[] __GPU_CPU__(float x) { return 1.0f / (x + 1e-12f); }, std::move(s))));
	}

	// Row-wise softmax: softmax along dim-1 (each row independently).
	// This is the axis attention needs: for a scores matrix indexed
	// [query, key], each query row becomes a distribution over keys.
	// Implemented as a column-softmax of the transpose.
	template <class D>
	Matrix<D> row_softmax(const Matrix<D>& M) {
		return col_softmax(M.T()).T();
	}

#ifdef CUDA
	// Row-wise softmax within each (seq_len x seq_len) attention block.
	// Input/output are stored (seq_len, seq_len * batchN) column-major; for each
	// (batch, query-row) pair we normalize across the seq_len keys of that block.
	__global__ void softmax_rows_batched_kernel(const float* x, float* y, int seq_len, int batchN) {
		int gid = blockIdx.x * blockDim.x + threadIdx.x;
		int total = seq_len * batchN;
		if (gid >= total) return;

		int blk = gid / seq_len;
		int a = gid % seq_len;            // query row within the block
		int col0 = blk * seq_len;

		float m = -1e30f;
		for (int b = 0; b < seq_len; ++b) {
			float v = x[a + (col0 + b) * seq_len];
			m = m > v ? m : v;
		}

		float s = 0.0f;
		for (int b = 0; b < seq_len; ++b) {
			float e = expf(x[a + (col0 + b) * seq_len] - m);
			y[a + (col0 + b) * seq_len] = e;
			s += e;
		}

		float inv = 1.0f / (s + 1e-12f);
		for (int b = 0; b < seq_len; ++b) {
			y[a + (col0 + b) * seq_len] *= inv;
		}
	}

	__global__ void causal_mask_kernel(float* s, int seq_len, int batchN, float mask_val) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int total = seq_len * seq_len * batchN;
		if (idx >= total) return;

		int local = idx % (seq_len * seq_len);
		int row = local % seq_len;
		int col = local / seq_len;

		// Causal masking: forbid attending to future keys (key index > query index).
		if (col > row) s[idx] = mask_val;
	}

	// Backward of the row-wise softmax.  A (row-normalized) is stored
	// (seq_len, seq_len*batchN) with A[a,b] at a + (blk*seq_len + b)*seq_len.
	// dAT holds d(A^T): dA[a,b] = dAT[b + (blk*seq_len + a)*seq_len].
	// One thread per (block, query-row a); reduces over keys b.
	__global__ void softmax_backward_rows_kernel(
		const float* A,
		const float* dAT,
		float* dS,
		int seq_len,
		int batchN,
		float scale) {
		int gid = blockIdx.x * blockDim.x + threadIdx.x;
		int total = seq_len * batchN;
		if (gid >= total) return;

		int blk = gid / seq_len;
		int a = gid % seq_len;            // query row
		int col0 = blk * seq_len;

		float row_sum = 0.0f;
		for (int b = 0; b < seq_len; ++b) {
			float av = A[a + (col0 + b) * seq_len];
			float da = dAT[b + (col0 + a) * seq_len];
			row_sum += av * da;
		}

		for (int b = 0; b < seq_len; ++b) {
			float av = A[a + (col0 + b) * seq_len];
			float da = dAT[b + (col0 + a) * seq_len];
			dS[a + (col0 + b) * seq_len] = av * (da - row_sum) * scale;
		}
	}

	inline void cuda_softmax_rows_batched(const float* x, float* y, int seq_len, int batchN) {
		const int total = seq_len * batchN;
		const int threads = 128;
		const int blocks = (total + threads - 1) / threads;
		softmax_rows_batched_kernel<<<blocks, threads>>>(x, y, seq_len, batchN);
		CudaErrorCheck(cudaGetLastError());
	}

	inline void cuda_apply_causal_mask(float* s, int seq_len, int batchN, float mask_val = -1e9f) {
		const int total = seq_len * seq_len * batchN;
		const int threads = 256;
		const int blocks = (total + threads - 1) / threads;
		causal_mask_kernel<<<blocks, threads>>>(s, seq_len, batchN, mask_val);
		CudaErrorCheck(cudaGetLastError());
	}

	inline void cuda_softmax_backward_rows(
		const float* A,
		const float* dAT,
		float* dS,
		int seq_len,
		int batchN,
		float scale) {
		const int total = seq_len * batchN;
		const int threads = 128;
		const int blocks = (total + threads - 1) / threads;
		softmax_backward_rows_kernel<<<blocks, threads>>>(
			A, dAT, dS, seq_len, batchN, scale);
		CudaErrorCheck(cudaGetLastError());
	}

	// Fused LayerNorm forward for one column c (x, y, xhat are (dim, N)
	// column-major, gamma/beta (dim,1)):
	//   y[:,c] = gamma ⊙ (x[:,c] - mu_c)/sqrt(var_c + eps) + beta
	// Also stores xhat and 1/sqrt(var+eps) for the backward pass. Replaces
	// ~8 generic matrix ops (and their temporaries) with one kernel.
	__global__ void layernorm_forward_kernel(
		const float* x, const float* gamma, const float* beta,
		float* y, float* xhat, float* inv_std, int dim, int N) {
		int c = blockIdx.x * blockDim.x + threadIdx.x;
		if (c >= N) return;
		const float* xc = x + (size_t)c * dim;

		float mu = 0.0f;
		for (int i = 0; i < dim; ++i) mu += xc[i];
		mu /= dim;

		float var = 0.0f;
		for (int i = 0; i < dim; ++i) {
			const float d = xc[i] - mu;
			var += d * d;
		}
		var /= dim;

		const float inv = rsqrtf(var + 1e-5f);
		inv_std[c] = inv;
		for (int i = 0; i < dim; ++i) {
			const float xh = (xc[i] - mu) * inv;
			xhat[(size_t)c * dim + i] = xh;
			y[(size_t)c * dim + i] = gamma[i] * xh + beta[i];
		}
	}

	// Fused LayerNorm input-gradient for one column (dxhat = dy ⊙ gamma):
	//   dx = inv_std ⊙ (dxhat - mean(dxhat) - xhat ⊙ mean(dxhat ⊙ xhat))
	__global__ void layernorm_backward_kernel(
		const float* dy, const float* gamma,
		const float* xhat, const float* inv_std,
		float* dx, int dim, int N) {
		int c = blockIdx.x * blockDim.x + threadIdx.x;
		if (c >= N) return;
		const size_t c0 = (size_t)c * dim;

		float m1 = 0.0f, m2 = 0.0f;
		for (int i = 0; i < dim; ++i) {
			const float dxh = gamma[i] * dy[c0 + i];
			m1 += dxh;
			m2 += dxh * xhat[c0 + i];
		}
		m1 /= dim;
		m2 /= dim;

		const float inv = inv_std[c];
		for (int i = 0; i < dim; ++i) {
			const float dxh = gamma[i] * dy[c0 + i];
			dx[c0 + i] = inv * (dxh - m1 - xhat[c0 + i] * m2);
		}
	}

	// y[i, c] += b[i] for every column: broadcast bias add, replacing the
	// "+ b * ones(1, N)" idiom (an outer-product GEMM plus a full temporary).
	__global__ void add_bias_kernel(float* y, const float* b, int rows, size_t total) {
		size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;
		y[idx] += b[idx % rows];
	}

	inline void cuda_layernorm_forward(
		const float* x, const float* gamma, const float* beta,
		float* y, float* xhat, float* inv_std, int dim, int N) {
		const int threads = 128;
		const int blocks = (N + threads - 1) / threads;
		layernorm_forward_kernel<<<blocks, threads>>>(x, gamma, beta, y, xhat, inv_std, dim, N);
		CudaErrorCheck(cudaGetLastError());
	}

	inline void cuda_layernorm_backward(
		const float* dy, const float* gamma,
		const float* xhat, const float* inv_std,
		float* dx, int dim, int N) {
		const int threads = 128;
		const int blocks = (N + threads - 1) / threads;
		layernorm_backward_kernel<<<blocks, threads>>>(dy, gamma, xhat, inv_std, dx, dim, N);
		CudaErrorCheck(cudaGetLastError());
	}

	inline void cuda_add_bias(float* y, const float* b, int rows, size_t total) {
		const int threads = 256;
		const int blocks = (int)((total + threads - 1) / threads);
		add_bias_kernel<<<blocks, threads>>>(y, b, rows, total);
		CudaErrorCheck(cudaGetLastError());
	}
#endif

	// Layer normalization over the feature dimension (rows) of each token/column.
	// x : (dim, N).  For each column c:
	//   y[:,c] = gamma ⊙ (x[:,c] - mean_c) / sqrt(var_c + eps) + beta
	// where mean/var are over the `dim` features. gamma,beta are (dim,1) learnable.
	// Implemented with backend-generic Matrix<D> ops so it works on CPU and GPU.
	template <class D>
	struct LayerNorm {
		int dim;
		Matrix<D> gamma, beta;
		adam_state<D> adam_g, adam_b;
		Matrix<D> ones_d1, ones_1N;
		Matrix<D> cached_xhat, cached_inv;   // xhat:(dim,N)  inv:(1,N)

		LayerNorm(int dim, int N)
			: dim(dim),
			  gamma(Matrix<D>::ones(dim, 1)), beta(Matrix<D>::zeros(dim, 1)),
			  adam_g(0.0001, dim, 1), adam_b(0.0001, dim, 1),
			  ones_d1("ln_d1", dim, 1), ones_1N("ln_1N", 1, N),
			  cached_xhat("ln_xhat", dim, N), cached_inv("ln_inv", 1, N) {
			ones_d1.ones();
			ones_1N.ones();
		}

		Matrix<D> forward(const Matrix<D>& x) {
#ifdef CUDA
			// Fused kernel (assumes plain column-major input, which is what
			// TransformerLayer always passes); other backends use the
			// generic matrix-op formulation below.
			if constexpr (std::is_same_v<D, CUDAfloat>) {
				if (!x.get_transpose()) {
					Matrix<D> y("ln_y", dim, x.num_col());
					cuda_layernorm_forward(
						reinterpret_cast<const float*>(x.data()),
						reinterpret_cast<const float*>(gamma.data()),
						reinterpret_cast<const float*>(beta.data()),
						const_cast<float*>(reinterpret_cast<const float*>(y.data())),
						const_cast<float*>(reinterpret_cast<const float*>(cached_xhat.data())),
						const_cast<float*>(reinterpret_cast<const float*>(cached_inv.data())),
						dim, (int)x.num_col());
					return y;
				}
			}
#endif
			const float invdim = 1.0f / (float)dim;
			auto mu  = sum(x, 0) * invdim;                       // (1,N) column means
			auto xc  = x - ones_d1 * mu;                         // (dim,N) centered
			auto var = sum(hadmd(xc, xc), 0) * invdim;           // (1,N)
			cached_inv = elemwise(
				[] __GPU_CPU__(float v) { return 1.0f / sqrtf(v + 1e-5f); }, std::move(var));
			cached_xhat = hadmd(xc, ones_d1 * cached_inv);       // (dim,N) standardized
			return hadmd(gamma * ones_1N, cached_xhat) + beta * ones_1N;
		}

		// dy : (dim,N) upstream gradient.  Returns dL/dx; updates gamma,beta if `update`.
		Matrix<D> backward(Matrix<D>&& dy, bool update) {
			Matrix<D> dx("ln_dx", dim, (int)dy.num_col());
			bool fused = false;
#ifdef CUDA
			if constexpr (std::is_same_v<D, CUDAfloat>) {
				if (!dy.get_transpose()) {
					cuda_layernorm_backward(
						reinterpret_cast<const float*>(dy.data()),
						reinterpret_cast<const float*>(gamma.data()),
						reinterpret_cast<const float*>(cached_xhat.data()),
						reinterpret_cast<const float*>(cached_inv.data()),
						const_cast<float*>(reinterpret_cast<const float*>(dx.data())),
						dim, (int)dy.num_col());
					fused = true;
				}
			}
#endif
			if (!fused) {
				const float invdim = 1.0f / (float)dim;
				auto dxhat = hadmd(dy, gamma * ones_1N);             // (dim,N)
				auto m1 = sum(dxhat, 0) * invdim;                    // (1,N)
				auto m2 = sum(hadmd(dxhat, cached_xhat), 0) * invdim;// (1,N)
				dx = hadmd(ones_d1 * cached_inv,
					dxhat - ones_d1 * m1 - hadmd(cached_xhat, ones_d1 * m2));
			}
			// Parameter gradients read gamma/xhat, so dx (above) comes first.
			if (update) {
				gamma -= adam_update(sum(hadmd(dy, cached_xhat), 1), adam_g);
				beta  -= adam_update(sum(dy, 1), adam_b);
			}
			return dx;
		}

		void set_gamma(const Matrix<D>& g) { gamma = g; }
		void set_beta(const Matrix<D>& b)  { beta = b; }
		const Matrix<D>& get_gamma() const { return gamma; }
		const Matrix<D>& get_beta() const  { return beta; }
	};

	//
	// Pre-LN multi-head self-attention + feed-forward block.
	//
	// Input  shape: (d_model, seq_len * batch)   — each column is one token.
	// Output shape: (d_model, seq_len * batch)   — same as input.
	//
	// Forward (pre-norm):
	//   1. x1 = LN1(x);  Q = Wq*x1,  K = Wk*x1,  V = Wv*x1
	//   2. Per head h and sequence i (d_h = d_k/num_heads):
	//        A = softmax( Q_ih^T K_ih / sqrt(d_h) , dim=keys )  (seq_len x seq_len)
	//        H_ih = V_ih * A^T
	//   3. R = x + (Wo * H + bo)                                (residual 1)
	//   4. x2 = LN2(R)
	//   5. out = R + (W2 * relu(W1*x2 + b1) + b2)               (residual 2)
	//
	// Weight dimensions:
	//   Wq, Wk, Wv : (d_k, d_model)          Wo : (d_model, d_k)
	//   bo         : (d_model, 1)
	//   W1         : (d_ff, d_model)          b1 : (d_ff, 1)
	//   W2         : (d_model, d_ff)          b2 : (d_model, 1)
	//   LN1, LN2   : gamma,beta each (d_model, 1)
	//
	// Template on D so the same code compiles for float, CUDAfloat, MPSfloat, etc.
	// All heavy lifting is done through the existing Matrix<D> BLAS/cuBLAS paths.

	template <class D>
	class TransformerLayer : public Layer<D> {
		int d_model, d_k, d_ff, seq_len, batchN;
		int num_heads, d_h;             // d_h = d_k / num_heads (per-head dim)
		bool causal;                    // apply causal (autoregressive) attention
		                                // mask when true; false = bidirectional
		                                // attention (e.g. for masked diffusion).

		// ── Attention weights ──
		Matrix<D> Wq, Wk, Wv, Wo, bo;
		adam_state<D> adam_Wq, adam_Wk, adam_Wv, adam_Wo, adam_bo;

		// ── FFN weights ──
		Matrix<D> W1, b1, W2, b2;
		adam_state<D> adam_W1, adam_b1, adam_W2, adam_b2;

		// ── Cached forward intermediates (needed for backward) ──
		Matrix<D> cached_Q, cached_K, cached_V;
		Matrix<D> cached_H;             // attention output before Wo
		Matrix<D> cached_R;             // residual after attention
		Matrix<D> cached_x1, cached_x2; // LN1(x) and LN2(R): inputs to attn / FFN
		Matrix<D> cached_F1;            // W1*x2 + b1  (pre-relu)
		Matrix<D> cached_F1_relu;       // relu(F1)

		// Per-sequence attention weight matrices: A_i = softmax(...)  shape (seq_len, seq_len)
		// Stored concatenated: (seq_len, seq_len * batchN)
		Matrix<D> cached_A;

		// Reusable buffers to avoid repeated allocation in hot loops.
		Matrix<D> ones_1N, ones_seq1, ones_1seq;
		Matrix<D> attn_scores_scratch, attn_dAiT_scratch, attn_dS_scratch;

		// Pre-norm LayerNorms: LN1 before attention, LN2 before FFN.
		LayerNorm<D> ln1, ln2;

		// y += b broadcast over columns. One kernel on CUDA; the generic
		// backends keep the outer-product formulation.
		void add_bias(Matrix<D>& y, const Matrix<D>& b) {
#ifdef CUDA
			if constexpr (std::is_same_v<D, CUDAfloat>) {
				if (!y.get_transpose()) {
					cuda_add_bias(
						const_cast<float*>(reinterpret_cast<const float*>(y.data())),
						reinterpret_cast<const float*>(b.data()),
						(int)y.num_row(), y.num_row() * y.num_col());
					return;
				}
			}
#endif
			y += b * ones_1N;
		}

	public:
		/**
		 * @param d_model  Token embedding dimension (input & output width).
		 * @param d_k      Total dimension of Q/K/V projections (across all heads).
		 * @param d_ff     Hidden dimension of the feed-forward sub-layer.
		 * @param seq_len  Number of tokens per sequence.
		 * @param batch    Number of sequences in a mini-batch.
		 * @param heads    Number of attention heads (must divide d_k). Each head
		 *                 attends independently over d_k/heads dims; Wo mixes them.
		 * @param causal   When true (default) apply the causal mask so each token
		 *                 attends only to itself and earlier tokens (autoregressive
		 *                 LM). When false, attention is bidirectional — every token
		 *                 sees the whole sequence, as required by a masked-diffusion
		 *                 denoiser.
		 */
		TransformerLayer(int d_model, int d_k, int d_ff, int seq_len, int batch, int heads = 1,
		                 bool causal = true)
			: Layer<D>(d_model, 1, seq_len * batch),
			  d_model(d_model), d_k(d_k), d_ff(d_ff),
			  seq_len(seq_len), batchN(batch),
			  num_heads(heads), d_h(d_k / heads), causal(causal),
			  // Attention weights
			  Wq(Matrix<D>::randn(d_k, d_model) * (1.0f / sqrtf((float)d_model))),
			  Wk(Matrix<D>::randn(d_k, d_model) * (1.0f / sqrtf((float)d_model))),
			  Wv(Matrix<D>::randn(d_k, d_model) * (1.0f / sqrtf((float)d_model))),
			  Wo(Matrix<D>::randn(d_model, d_k) * (1.0f / sqrtf((float)d_k))),
			  bo(Matrix<D>::zeros(d_model, 1)),
			  adam_Wq(0.0001, d_k, d_model), adam_Wk(0.0001, d_k, d_model),
			  adam_Wv(0.0001, d_k, d_model), adam_Wo(0.0001, d_model, d_k),
			  adam_bo(0.0001, d_model, 1),
			  // FFN weights
			  W1(Matrix<D>::randn(d_ff, d_model) * (1.0f / sqrtf((float)d_model))),
			  b1(Matrix<D>::zeros(d_ff, 1)),
			  W2(Matrix<D>::randn(d_model, d_ff) * (1.0f / sqrtf((float)d_ff))),
			  b2(Matrix<D>::zeros(d_model, 1)),
			  adam_W1(0.0001, d_ff, d_model), adam_b1(0.0001, d_ff, 1),
			  adam_W2(0.0001, d_model, d_ff), adam_b2(0.0001, d_model, 1),
			  // Cached tensors (fixed-size for this layer instance)
			  cached_Q("tq", d_k, seq_len * batch), cached_K("tk", d_k, seq_len * batch),
			  cached_V("tv", d_k, seq_len * batch), cached_H("th", d_k, seq_len * batch),
			  cached_R("tr", d_model, seq_len * batch),
			  cached_x1("tx1", d_model, seq_len * batch), cached_x2("tx2", d_model, seq_len * batch),
			  cached_F1("tf1", d_ff, seq_len * batch), cached_F1_relu("tf1r", d_ff, seq_len * batch),
			  cached_A("ta", seq_len, seq_len * batch * heads),
			  ones_1N("ones_1N", 1, seq_len * batch), ones_seq1("ones_seq1", seq_len, 1),
			  ones_1seq("ones_1seq", 1, seq_len),
			  attn_scores_scratch("attn_scores", seq_len, seq_len * batch * heads),
			  attn_dAiT_scratch("attn_dAiT", seq_len, seq_len * batch * heads),
			  attn_dS_scratch("attn_dS", seq_len, seq_len * batch * heads),
			  ln1(d_model, seq_len * batch), ln2(d_model, seq_len * batch)
		{
			if (heads <= 0 || d_k % heads != 0) {
				LOG_ERROR("TransformerLayer: num_heads ({}) must be a positive divisor of d_k ({})", heads, d_k);
				ERROR_OUT;
			}
			// Override base-class weights/bias/adam to be dummy (unused).
			this->weights = Matrix<D>::zeros(1, 1);
			this->bias    = Matrix<D>::zeros(1, 1);
			this->adamW   = adam_state<D>(0.0001, 1, 1);
			this->adamb   = adam_state<D>(0.0001, 1, 1);
			ones_1N.ones();
			ones_seq1.ones();
			ones_1seq.ones();
		}

		void eval(const Matrix<D>& input) override {
			const int N = seq_len * batchN;

			// 1. LN1 then linear projections  — Q,K,V : (d_k, N)
			cached_x1 = ln1.forward(input);
			cached_Q = Wq * cached_x1;
			cached_K = Wk * cached_x1;
			cached_V = Wv * cached_x1;

			// 2. Multi-head self-attention. Each head attends over its own d_h
			//    rows of Q/K/V independently; the concatenated heads (= cached_H,
			//    d_k rows) are mixed by Wo below. Attention blocks are stored
			//    head-major: block (head hh, sequence b) lives at column
			//    (hh*batchN + b)*seq_len in the (seq_len, seq_len*batchN*num_heads)
			//    scratch/cached_A buffers.
			const float scale = 1.0f / sqrtf((float)d_h);

#ifdef CUDA
			if constexpr (std::is_same_v<D, CUDAfloat>) {
				const float zero = 0.0f, one = 1.0f;
				const long long stride_qkv = (long long)d_k * (long long)seq_len;
				const long long stride_attn = (long long)seq_len * (long long)seq_len;
				const long long head_attn = stride_attn * (long long)batchN;

				const float* q_ptr = reinterpret_cast<const float*>(cached_Q.data());
				const float* k_ptr = reinterpret_cast<const float*>(cached_K.data());
				const float* v_ptr = reinterpret_cast<const float*>(cached_V.data());
				float* scores_ptr = const_cast<float*>(reinterpret_cast<const float*>(attn_scores_scratch.data()));
				const float* a_ptr = reinterpret_cast<const float*>(cached_A.data());
				float* h_ptr = const_cast<float*>(reinterpret_cast<const float*>(cached_H.data()));

				// scores_h = (Q_h^T K_h) * scale  for each head's d_h-row slice.
				for (int hh = 0; hh < num_heads; ++hh) {
					CuBLASErrorCheck(cublasSgemmStridedBatched(
						Matrix<CUDAfloat>::global_handle,
						CUBLAS_OP_T, CUBLAS_OP_N,
						seq_len, seq_len, d_h,
						&scale,
						q_ptr + hh * d_h, d_k, stride_qkv,
						k_ptr + hh * d_h, d_k, stride_qkv,
						&zero,
						scores_ptr + hh * head_attn, seq_len, stride_attn,
						batchN));
				}

				if (causal)
					cuda_apply_causal_mask(scores_ptr, seq_len, batchN * num_heads);

				cuda_softmax_rows_batched(scores_ptr,
					const_cast<float*>(reinterpret_cast<const float*>(cached_A.data())),
					seq_len, batchN * num_heads);

				// H_h = V_h * A_h^T  written into head hh's d_h rows of cached_H.
				for (int hh = 0; hh < num_heads; ++hh) {
					CuBLASErrorCheck(cublasSgemmStridedBatched(
						Matrix<CUDAfloat>::global_handle,
						CUBLAS_OP_N, CUBLAS_OP_T,
						d_h, seq_len, seq_len,
						&one,
						v_ptr + hh * d_h, d_k, stride_qkv,
						a_ptr + hh * head_attn, seq_len, stride_attn,
						&zero,
						h_ptr + hh * d_h, d_k, stride_qkv,
						batchN));
				}
			} else
#endif
			{
				for (int hh = 0; hh < num_heads; ++hh) {
					const int r0 = hh * d_h;
					for (int b = 0; b < batchN; ++b) {
						const int c0 = b * seq_len;
						// Per-head Q_i, K_i, V_i : (d_h, seq_len)
						auto Qi = cached_Q.slice(r0, r0 + d_h, c0, c0 + seq_len);
						auto Ki = cached_K.slice(r0, r0 + d_h, c0, c0 + seq_len);
						auto Vi = cached_V.slice(r0, r0 + d_h, c0, c0 + seq_len);

						// scores : (seq_len, seq_len) = Qi^T * Ki * scale
						auto scores = Qi.T() * Ki * scale;
						// Causal masking: for query row, disallow future keys col > row.
						if (causal) {
#ifdef APPLE_SILICON
							// `scores` is an async GPU (Metal) matmul result. The
							// mask below writes it via host elem(); without first
							// waiting for the GEMM, the in-flight kernel overwrites
							// those host writes and the mask is lost.
							mpsSynchronize();
#endif
							for (int row = 0; row < seq_len; ++row) {
								for (int col = row + 1; col < seq_len; ++col) {
									scores.elem(row, col) = -1e9f;
								}
							}
						}
						// A_i : (seq_len, seq_len) — softmax each row (over keys)
						auto Ai = row_softmax(scores);

						// H_i : (d_h, seq_len) = Vi * Ai.T
						auto Hi = Vi * Ai.T();

						const int ablk = hh * batchN + b;
						cached_H.slice(r0, r0 + d_h, c0, c0 + seq_len, Hi);
						cached_A.slice(0, seq_len, ablk * seq_len, (ablk + 1) * seq_len, Ai);
					}
				}
			}

			// 3. Output projection + residual 1
			auto O = Wo * cached_H;
			add_bias(O, bo);
			cached_R = input + O;

			// 4. LN2 then FFN: relu(W1*x2 + b1) then W2*...+b2
			cached_x2 = ln2.forward(cached_R);
			cached_F1 = W1 * cached_x2;
			add_bias(cached_F1, b1);
			cached_F1_relu = relu(Matrix<D>(cached_F1));
			auto F = W2 * cached_F1_relu;
			add_bias(F, b2);

			// 5. Residual 2
			this->val = cached_R + F;
		}

		Matrix<D> grad(const Matrix<D>&) const override {
			return Matrix<D>::ones(1, 1);
		}

		Matrix<D> backward(const Matrix<D>& input, Matrix<D>&& upstream_grad) override {
			const int N = seq_len * batchN;
			auto dout = std::move(upstream_grad);         // (d_model, N)

			// ── FFN backward ──────────────────────────────────────────────
			// dout doubles as dF (from second residual); it is only read here.
			auto dW2 = dout * cached_F1_relu.T();          // (d_model, d_ff)
			auto db2 = sum(dout, 1);                        // (d_model, 1)
			auto dF1_relu = W2.T() * dout;                  // (d_ff, N)
			auto dF1 = hadmd(d_relu(Matrix<D>(cached_F1)), std::move(dF1_relu)); // (d_ff, N)
			auto dW1 = dF1 * cached_x2.T();                // (d_ff, d_model)
			auto db1 = sum(dF1, 1);                         // (d_ff, 1)
			auto dx2 = W1.T() * dF1;                        // (d_model, N) — grad wrt LN2 output
			// Back through LN2, then add residual-1 path (out = R + F).
			auto dR = ln2.backward(std::move(dx2), this->need_update) + dout;

			// ── Attention output projection backward ──────────────────────
			// O = Wo * H + bo  →  dR is dO (read-only here); it also flows
			// through the residual into dx at the end of this function.
			auto dWo = dR * cached_H.T();                   // (d_model, d_k)
			auto dbo = sum(dR, 1);                           // (d_model, 1)
			auto dH = Wo.T() * dR;                           // (d_k, N)

			// ── Self-attention backward (per sequence) ────────────────────
			// Written in full below (GEMM beta=0 on CUDA, per-slice
			// assignment otherwise), so no zero-initialization is needed.
			Matrix<D> dQ("dQ", d_k, N), dK("dK", d_k, N), dV("dV", d_k, N);
			const float scale = 1.0f / sqrtf((float)d_h);

#ifdef CUDA
			if constexpr (std::is_same_v<D, CUDAfloat>) {
				const float one = 1.0f;
				const float zero = 0.0f;
				const long long stride_qkv = (long long)d_k * (long long)seq_len;
				const long long stride_attn = (long long)seq_len * (long long)seq_len;
				const long long head_attn = stride_attn * (long long)batchN;

				const float* dh_ptr = reinterpret_cast<const float*>(dH.data());
				const float* a_ptr = reinterpret_cast<const float*>(cached_A.data());
				const float* v_ptr = reinterpret_cast<const float*>(cached_V.data());
				const float* k_ptr = reinterpret_cast<const float*>(cached_K.data());
				const float* q_ptr = reinterpret_cast<const float*>(cached_Q.data());
				float* dv_ptr = const_cast<float*>(reinterpret_cast<const float*>(dV.data()));
				float* dait_ptr = const_cast<float*>(reinterpret_cast<const float*>(attn_dAiT_scratch.data()));
				const float* ds_ptr = reinterpret_cast<const float*>(attn_dS_scratch.data());
				float* dq_ptr = const_cast<float*>(reinterpret_cast<const float*>(dQ.data()));
				float* dk_ptr = const_cast<float*>(reinterpret_cast<const float*>(dK.data()));

				// dV_h = dH_h * A_h ;  dA_h^T = V_h^T * dH_h   (per head)
				for (int hh = 0; hh < num_heads; ++hh) {
					CuBLASErrorCheck(cublasSgemmStridedBatched(
						Matrix<CUDAfloat>::global_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						d_h, seq_len, seq_len,
						&one,
						dh_ptr + hh * d_h, d_k, stride_qkv,
						a_ptr + hh * head_attn, seq_len, stride_attn,
						&zero,
						dv_ptr + hh * d_h, d_k, stride_qkv,
						batchN));

					CuBLASErrorCheck(cublasSgemmStridedBatched(
						Matrix<CUDAfloat>::global_handle,
						CUBLAS_OP_T, CUBLAS_OP_N,
						seq_len, seq_len, d_h,
						&one,
						v_ptr + hh * d_h, d_k, stride_qkv,
						dh_ptr + hh * d_h, d_k, stride_qkv,
						&zero,
						dait_ptr + hh * head_attn, seq_len, stride_attn,
						batchN));
				}

				cuda_softmax_backward_rows(
					reinterpret_cast<const float*>(cached_A.data()),
					reinterpret_cast<const float*>(attn_dAiT_scratch.data()),
					const_cast<float*>(reinterpret_cast<const float*>(attn_dS_scratch.data())),
					seq_len,
					batchN * num_heads,
					scale);

				// dQ_h = K_h * dS_h^T ;  dK_h = Q_h * dS_h   (per head)
				for (int hh = 0; hh < num_heads; ++hh) {
					CuBLASErrorCheck(cublasSgemmStridedBatched(
						Matrix<CUDAfloat>::global_handle,
						CUBLAS_OP_N, CUBLAS_OP_T,
						d_h, seq_len, seq_len,
						&one,
						k_ptr + hh * d_h, d_k, stride_qkv,
						ds_ptr + hh * head_attn, seq_len, stride_attn,
						&zero,
						dq_ptr + hh * d_h, d_k, stride_qkv,
						batchN));

					CuBLASErrorCheck(cublasSgemmStridedBatched(
						Matrix<CUDAfloat>::global_handle,
						CUBLAS_OP_N, CUBLAS_OP_N,
						d_h, seq_len, seq_len,
						&one,
						q_ptr + hh * d_h, d_k, stride_qkv,
						ds_ptr + hh * head_attn, seq_len, stride_attn,
						&zero,
						dk_ptr + hh * d_h, d_k, stride_qkv,
						batchN));
				}
			} else
#endif
			{
				for (int hh = 0; hh < num_heads; ++hh) {
					const int r0 = hh * d_h;
					for (int b = 0; b < batchN; ++b) {
						const int c0 = b * seq_len;
						const int ablk = hh * batchN + b;
						auto Qi = cached_Q.slice(r0, r0 + d_h, c0, c0 + seq_len);
						auto Ki = cached_K.slice(r0, r0 + d_h, c0, c0 + seq_len);
						auto Vi = cached_V.slice(r0, r0 + d_h, c0, c0 + seq_len);
						auto Ai = cached_A.slice(0, seq_len, ablk * seq_len, (ablk + 1) * seq_len);
						auto dHi = dH.slice(r0, r0 + d_h, c0, c0 + seq_len);

						// H_i = V_i * A_i^T
						// dV_i = dH_i * A_i            (d_h, seq_len)
						// dA_i^T = V_i^T * dH_i        (seq_len, seq_len)
						auto dVi = dHi * Ai;
						auto dAiT = Vi.T() * dHi;       // (seq_len, seq_len) — gradient of A^T
						auto dAi = dAiT.T();             // gradient of A itself

						// Softmax backward (row-wise softmax over keys):
						// A_i = softmax(S_i, dim=1) where S_i = Q_i^T K_i * scale
						// dS = A ⊙ (dA − (sum(A ⊙ dA, over keys))·1ᵀ)
						auto AodA = hadmd(Ai, dAi);                              // (seq_len, seq_len)
						auto row_sum = sum(AodA, 1);                             // (seq_len, 1)
						auto dSi = hadmd(Ai, dAi - row_sum * ones_1seq) * scale;
						// S_i = Q_i^T * K_i * scale
						// dQ_i = K_i * dS_i^T          (d_h, seq_len)
						// dK_i = Q_i * dS_i            (d_h, seq_len)
						auto dQi = Ki * dSi.T();
						auto dKi = Qi * dSi;

						dQ.slice(r0, r0 + d_h, c0, c0 + seq_len, dQi);
						dK.slice(r0, r0 + d_h, c0, c0 + seq_len, dKi);
						dV.slice(r0, r0 + d_h, c0, c0 + seq_len, dVi);
					}
				}
			}

			// Projection backward: Q = Wq*x1, K = Wk*x1, V = Wv*x1  (x1 = LN1(input))
			auto dWq = dQ * cached_x1.T();
			auto dWk = dK * cached_x1.T();
			auto dWv = dV * cached_x1.T();

			// Grad wrt LN1 output, back through LN1, plus the residual path
			// (R = x + attention output, so dR flows straight into dx).
			auto dx1 = Wq.T() * dQ + Wk.T() * dK + Wv.T() * dV;
			auto dx = ln1.backward(std::move(dx1), this->need_update) + dR;

			// ── Parameter updates ─────────────────────────────────────────
			if (this->need_update) {
				Wq -= adam_update(std::move(dWq), adam_Wq);
				Wk -= adam_update(std::move(dWk), adam_Wk);
				Wv -= adam_update(std::move(dWv), adam_Wv);
				Wo -= adam_update(std::move(dWo), adam_Wo);
				bo -= adam_update(std::move(dbo), adam_bo);
				W1 -= adam_update(std::move(dW1), adam_W1);
				b1 -= adam_update(std::move(db1), adam_b1);
				W2 -= adam_update(std::move(dW2), adam_W2);
				b2 -= adam_update(std::move(db2), adam_b2);
			}

			return dx;
		}

		// ── Accessors for parity testing ──
		void set_Wq(const Matrix<D>& w) { Wq = w; }
		void set_Wk(const Matrix<D>& w) { Wk = w; }
		void set_Wv(const Matrix<D>& w) { Wv = w; }
		void set_Wo(const Matrix<D>& w) { Wo = w; }
		void set_bo(const Matrix<D>& w) { bo = w; }
		void set_W1(const Matrix<D>& w) { W1 = w; }
		void set_b1(const Matrix<D>& w) { b1 = w; }
		void set_W2(const Matrix<D>& w) { W2 = w; }
		void set_b2(const Matrix<D>& w) { b2 = w; }

		const Matrix<D>& get_Wq() const { return Wq; }
		const Matrix<D>& get_Wk() const { return Wk; }
		const Matrix<D>& get_Wv() const { return Wv; }
		const Matrix<D>& get_Wo() const { return Wo; }
		const Matrix<D>& get_bo() const { return bo; }
		const Matrix<D>& get_W1() const { return W1; }
		const Matrix<D>& get_b1() const { return b1; }
		const Matrix<D>& get_W2() const { return W2; }
		const Matrix<D>& get_b2() const { return b2; }
		int heads() const { return num_heads; }

		void set_ln1_gamma(const Matrix<D>& v) { ln1.set_gamma(v); }
		void set_ln1_beta(const Matrix<D>& v)  { ln1.set_beta(v); }
		void set_ln2_gamma(const Matrix<D>& v) { ln2.set_gamma(v); }
		void set_ln2_beta(const Matrix<D>& v)  { ln2.set_beta(v); }
		const Matrix<D>& get_ln1_gamma() const { return ln1.get_gamma(); }
		const Matrix<D>& get_ln1_beta() const  { return ln1.get_beta(); }
		const Matrix<D>& get_ln2_gamma() const { return ln2.get_gamma(); }
		const Matrix<D>& get_ln2_beta() const  { return ln2.get_beta(); }

		// Set the Adam learning rate for every parameter group in this block
		// (attention, FFN, and both LayerNorms). Used to drive an LR schedule.
		void set_lr(float lr) {
			adam_Wq.alpha = lr; adam_Wk.alpha = lr; adam_Wv.alpha = lr;
			adam_Wo.alpha = lr; adam_bo.alpha = lr;
			adam_W1.alpha = lr; adam_b1.alpha = lr;
			adam_W2.alpha = lr; adam_b2.alpha = lr;
			ln1.adam_g.alpha = lr; ln1.adam_b.alpha = lr;
			ln2.adam_g.alpha = lr; ln2.adam_b.alpha = lr;
		}
	};

}

#endif
