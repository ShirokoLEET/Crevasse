#include "d3d_cuda_interop.h"

#include <cuda_d3d11_interop.h>
#include <dxgi.h>

bool register_d3d11_texture(ID3D11Texture2D *tex, D3D11CudaInterop &out)
{
	if (!tex)
		return false;

	// Get source texture dimensions
	D3D11_TEXTURE2D_DESC desc;
	tex->GetDesc(&desc);

	// Check if staging texture size matches
	bool need_new_staging = !out.staging_tex || out.width != (int)desc.Width || out.height != (int)desc.Height;

	if (need_new_staging) {
		// Unregister old CUDA resource
		if (out.cuda_res) {
			cudaGraphicsUnregisterResource(out.cuda_res);
			out.cuda_res = nullptr;
		}

		// Release old staging texture
		if (out.staging_tex) {
			out.staging_tex->Release();
			out.staging_tex = nullptr;
		}

		// Create new staging texture with DEFAULT usage (required for CUDA interop)
		ID3D11Device *dev;
		tex->GetDevice(&dev);

		D3D11_TEXTURE2D_DESC staging_desc = {};
		staging_desc.Width = desc.Width;
		staging_desc.Height = desc.Height;
		staging_desc.MipLevels = 1;
		staging_desc.ArraySize = 1;
		staging_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		staging_desc.SampleDesc.Count = 1;
		staging_desc.Usage = D3D11_USAGE_DEFAULT;
		staging_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		staging_desc.MiscFlags = 0; // No special flags needed for same-device CUDA interop

		HRESULT hr = dev->CreateTexture2D(&staging_desc, nullptr, &out.staging_tex);
		dev->Release();

		if (FAILED(hr)) {
			out.staging_tex = nullptr;
			out.width = 0;
			out.height = 0;
			return false;
		}

		// Register the staging texture with CUDA
		const cudaError_t err = cudaGraphicsD3D11RegisterResource(&out.cuda_res, out.staging_tex,
									  cudaGraphicsRegisterFlagsNone);

		if (err != cudaSuccess) {
			out.staging_tex->Release();
			out.staging_tex = nullptr;
			out.cuda_res = nullptr;
			out.width = 0;
			out.height = 0;
			return false;
		}

		out.width = (int)desc.Width;
		out.height = (int)desc.Height;
	}

	// Copy source texture to staging texture
	ID3D11Device *dev;
	tex->GetDevice(&dev);
	ID3D11DeviceContext *ctx;
	dev->GetImmediateContext(&ctx);

	ctx->CopyResource(out.staging_tex, tex);
	ctx->Flush(); // Ensure copy completes before CUDA access

	ctx->Release();
	dev->Release();

	out.texture = tex;
	return true;
}

void unregister_d3d11_texture(D3D11CudaInterop &interop)
{
	if (interop.cuda_res) {
		cudaGraphicsUnregisterResource(interop.cuda_res);
		interop.cuda_res = nullptr;
	}
	if (interop.staging_tex) {
		interop.staging_tex->Release();
		interop.staging_tex = nullptr;
	}
	interop.texture = nullptr;
	interop.width = 0;
	interop.height = 0;
}

static bool g_cuda_d3d11_initialized = false;

bool init_cuda_d3d11_device(ID3D11Device *d3d_device)
{
	if (g_cuda_d3d11_initialized)
		return true;

	if (!d3d_device)
		return false;

	// Get the DXGI device from D3D11 device
	IDXGIDevice *dxgi_device = nullptr;
	HRESULT hr = d3d_device->QueryInterface(__uuidof(IDXGIDevice), (void **)&dxgi_device);
	if (FAILED(hr) || !dxgi_device)
		return false;

	// Get the adapter from DXGI device
	IDXGIAdapter *adapter = nullptr;
	hr = dxgi_device->GetAdapter(&adapter);
	dxgi_device->Release();

	if (FAILED(hr) || !adapter)
		return false;

	// Find the CUDA device that corresponds to this adapter
	int cuda_device = -1;
	cudaError_t err = cudaD3D11GetDevice(&cuda_device, adapter);
	adapter->Release();

	if (err != cudaSuccess || cuda_device < 0) {
		return false;
	}

	// Set the CUDA device
	err = cudaSetDevice(cuda_device);
	if (err != cudaSuccess) {
		return false;
	}

	g_cuda_d3d11_initialized = true;
	return true;
}