#pragma once

#include <d3d11.h>

struct cudaGraphicsResource;

struct D3D11CudaInterop {
	struct cudaGraphicsResource *cuda_res = nullptr;
	ID3D11Texture2D *texture = nullptr;
	int width = 0;
	int height = 0;
};

/**
 * @brief Registers a D3D11 texture for access by CUDA.
 * 
 * @param tex The Direct3D 11 texture to register.
 * @param out The interop structure to store the registration handle.
 * @return true if registration was successful, false otherwise.
 */
bool register_d3d11_texture(ID3D11Texture2D *tex, D3D11CudaInterop &out);