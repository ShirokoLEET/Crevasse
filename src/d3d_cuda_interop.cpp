#include "d3d_cuda_interop.h"

#include <cuda_d3d11_interop.h>

bool register_d3d11_texture(ID3D11Texture2D *tex, D3D11CudaInterop &out)
{
	if (!tex)
		return false;

	if (out.texture == tex && out.cuda_res)
		return true;

	if (out.cuda_res) {
		cudaGraphicsUnregisterResource(out.cuda_res);
		out.cuda_res = nullptr;
		out.texture = nullptr;
	}

	const cudaError_t err = cudaGraphicsD3D11RegisterResource(&out.cuda_res, tex, cudaGraphicsRegisterFlagsNone);

	if (err != cudaSuccess) {
		out.cuda_res = nullptr;
		out.texture = nullptr;
		return false;
	}

	out.texture = tex;
	return true;
}