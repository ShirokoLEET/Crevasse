#include "infer.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_d3d11_interop.h>
#include "logutils.cpp"

// Declare the preprocessing function
extern "C" int run_preprocess(cudaTextureObject_t texObj, float *d_output, int input_w, int input_h, int target_w,
			      int target_h, cudaStream_t stream);

class infer::Logger : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char *msg) noexcept override
	{
		// Suppress info/verbose logs if needed, or route to OBS logs
		if (severity != Severity::kINFO && severity != Severity::kVERBOSE) {
			log_info(msg);
		}
	}
};

infer::infer()
{
	logger = new Logger();
	cudaStreamCreate(&stream);
}

infer::~infer()
{
	if (context)
		delete context;
	if (engine)
		delete engine;
	if (runtime)
		delete runtime;
	if (logger)
		delete logger;

	if (d_input)
		cudaFree(d_input);
	if (d_output)
		cudaFree(d_output);

	if (stream)
		cudaStreamDestroy(stream);
}

bool infer::init(const std::string &model_path)
{
	std::ifstream file(model_path, std::ios::binary);
	if (!file.good()) {
		log_info(model_path.c_str());
		return false;
	}

	file.seekg(0, file.end);
	size_t size = file.tellg();
	file.seekg(0, file.beg);

	std::vector<char> trtModelStream(size);
	file.read(trtModelStream.data(), size);
	file.close();

	runtime = nvinfer1::createInferRuntime(*logger);
	if (!runtime)
		return false;

	engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
	if (!engine)
		return false;

	context = engine->createExecutionContext();
	if (!context)
		return false;

	// Allocate buffers
	// Input: 1x3x640x640
	size_t input_bytes = input_w * input_h * input_c * sizeof(float);
	if (cudaMalloc(&d_input, input_bytes) != cudaSuccess)
		return false;

	// Output: 1x5x8400
	size_t output_bytes = output_size * sizeof(float);
	if (cudaMalloc(&d_output, output_bytes) != cudaSuccess)
		return false;

	return true;
}

bool infer::forward(D3D11CudaInterop &interop, float *host_output)
{
	if (!context || !interop.cuda_res)
		return false;

	// Map D3D resource
	cudaError_t err = cudaGraphicsMapResources(1, &interop.cuda_res, stream);
	if (err != cudaSuccess) {
		log_info("cudaGraphicsMapResources failed");
		log_info(cudaGetErrorString(err));
		return false;
	}

	cudaArray_t cuArray;
	err = cudaGraphicsSubResourceGetMappedArray(&cuArray, interop.cuda_res, 0, 0);
	if (err != cudaSuccess) {
		cudaGraphicsUnmapResources(1, &interop.cuda_res, stream);
		return false;
	}

	// Create Texture Object
	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc = {};
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType; // Read raw values (uchar4 for 8-bit)
	texDesc.normalizedCoords = 0;

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	// Run Preprocess
	run_preprocess(texObj, (float *)d_input, interop.width, interop.height, input_w, input_h, stream);

	cudaDestroyTextureObject(texObj);
	cudaGraphicsUnmapResources(1, &interop.cuda_res, stream);

	if (!context->setTensorAddress("images", d_input)) {
		log_info("Failed to set input tensor address: images");
		return false;
	}

	if (!context->setTensorAddress("output0", d_output)) {
		log_info("Failed to set output tensor address: output0");
		return false;
	}
	if (!context->enqueueV3(stream)) {
		log_info("Failed to enqueue inference");
		return false;
	}

	if (host_output) {
		cudaMemcpyAsync(host_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	}

	cudaStreamSynchronize(stream);
	return true;
}
