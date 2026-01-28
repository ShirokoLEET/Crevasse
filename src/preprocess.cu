#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <stdio.h>

// CUDA Kernel for center crop and normalization
// Crops a target_width x target_height region from the exact center of the input image.
// Output is NCHW (1 x 3 x Height x Width) Planar RGB float [0.0, 1.0]
__global__ void center_crop_kernel(cudaTextureObject_t texObj, float *output, int input_width, int input_height,
				   int target_width, int target_height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= target_width || y >= target_height)
		return;

	// Calculate offset to place crop region at the center of input
	// For 2560x1440 input and 640x640 target:
	//   offset_x = (2560 - 640) / 2 = 960  -> crop from x: 960 to 1599
	//   offset_y = (1440 - 640) / 2 = 400  -> crop from y: 400 to 1039
	int offset_x = (input_width - target_width) / 2;
	int offset_y = (input_height - target_height) / 2;

	// Fetch pixel at (offset_x + x, offset_y + y)
	uchar4 pixel = tex2D<uchar4>(texObj, (float)(offset_x + x), (float)(offset_y + y));

	// D3D11 DXGI_FORMAT_B8G8R8A8_UNORM: x=Blue, y=Green, z=Red
	float r = (float)pixel.z / 255.0f;
	float g = (float)pixel.y / 255.0f;
	float b = (float)pixel.x / 255.0f;

	int area = target_width * target_height;
	int idx = y * target_width + x;

	// Write to planar RGB output (NCHW format)
	output[idx] = r;            // R channel
	output[area + idx] = g;     // G channel
	output[2 * area + idx] = b; // B channel
}

// Host wrapper function
// Returns 0 on success
extern "C" int run_preprocess(cudaTextureObject_t texObj, float *d_output, int input_w, int input_h, int target_w,
			      int target_h, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid((target_w + block.x - 1) / block.x, (target_h + block.y - 1) / block.y);

	center_crop_kernel<<<grid, block, 0, stream>>>(texObj, d_output, input_w, input_h, target_w, target_h);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Error in preprocess: %s\n", cudaGetErrorString(err));
		return -1;
	}
	return 0;
}
