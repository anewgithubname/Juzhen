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

#ifdef CUDA
#include "../cpp/cumatrix.cuh"
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

#if !defined(CUDA) && !defined(APPLE_SILICON)
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
#endif // !CUDA && !APPLE_SILICON

}

#endif
