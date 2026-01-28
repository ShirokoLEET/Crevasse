#pragma once

#include <d3d11.h>

struct cudaGraphicsResource;

struct D3D11CudaInterop {
	struct cudaGraphicsResource *cuda_res = nullptr;
	ID3D11Texture2D *texture = nullptr;     // The registered texture
	ID3D11Texture2D *staging_tex = nullptr; // Owned staging texture for CUDA
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

/**
 * @brief Unregisters a D3D11 texture from CUDA access.
 * 
 * @param interop The interop structure containing the registration handle.
 */
void unregister_d3d11_texture(D3D11CudaInterop &interop);

/**
 * @brief Initializes CUDA to use the same device as D3D11.
 * Must be called before using CUDA-D3D11 interop.
 * 
 * @param d3d_device The D3D11 device to associate with CUDA.
 * @return true if initialization was successful, false otherwise.
 */
bool init_cuda_d3d11_device(ID3D11Device *d3d_device);