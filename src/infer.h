#pragma once
#include <string>
#include <vector>
#include <memory>
#include <d3d11.h>
#include <cuda_runtime.h>
#include "d3d_cuda_interop.h"

// Forward declare TensorRT classes to avoid exposing NvInfer headers here unless necessary
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
class ILogger;
} // namespace nvinfer1

class infer {
public:
	infer();
	~infer();

	bool init(const std::string &model_path);
	bool forward(D3D11CudaInterop &interop, float *output_data);

private:
	struct InferImpl;
	class Logger; // Custom logger

	nvinfer1::IRuntime *runtime = nullptr;
	nvinfer1::ICudaEngine *engine = nullptr;
	nvinfer1::IExecutionContext *context = nullptr;
	Logger *logger = nullptr;

	void *d_input = nullptr;
	void *d_output = nullptr;

	const int input_w = 640;
	const int input_h = 640;
	const int input_c = 3;
	const int output_size = 1 * 5 * 8400; // As per user request

	cudaStream_t stream = nullptr;
};
