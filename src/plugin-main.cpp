#include <obs-module.h>
#include <plugin-support.h>
#include <graphics/graphics.h>
#include <d3d11.h>
#include "d3d_cuda_interop.h"
#include <cuda_runtime.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

struct crevasse_filter {
	obs_source_t *source;
	void *d3d11_tex = nullptr;
	ID3D11Texture2D *copy_tex = nullptr;
	uint32_t width = 0;
	uint32_t height = 0;
};

static bool inited_d3d11_cuda = false;
static D3D11CudaInterop g_interop = {};

static const char *filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Crevasse_ai";
}

static void filter_destroy(void *data)
{
	auto *tf = static_cast<crevasse_filter *>(data);
	if (!tf)
		return;

	if (tf->copy_tex) {
		tf->copy_tex->Release();
		tf->copy_tex = nullptr;
	}

	bfree(tf);
}

static void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	auto *tf = static_cast<crevasse_filter *>(bzalloc(sizeof(crevasse_filter)));

	tf->source = source;

	UNUSED_PARAMETER(settings);
	return tf;
}

static void filter_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);

	auto *tf = static_cast<crevasse_filter *>(data);
	if (!tf)
		return;

	obs_source_t *target = obs_filter_get_target(tf->source);

	if (!obs_source_process_filter_begin(tf->source, GS_RGBA, OBS_NO_DIRECT_RENDERING)) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	gs_texture_t *rt = gs_get_render_target();
	ID3D11Texture2D *d3d_tex = rt ? static_cast<ID3D11Texture2D *>(gs_texture_get_obj(rt)) : NULL;
	ID3D11Texture2D *old_tex = static_cast<ID3D11Texture2D *>(tf->d3d11_tex);
	const bool tex_changed = (d3d_tex != old_tex);

	if (tex_changed) {
		blog(LOG_INFO, "[crevasse] D3D11 texture changed: %p -> %p", old_tex, d3d_tex);
		tf->d3d11_tex = d3d_tex;
	}

	if (!d3d_tex) {
		obs_log(LOG_ERROR, "[crevasse] No D3D11 render target texture available yet");
	} else {
		D3D11_TEXTURE2D_DESC desc;
		d3d_tex->GetDesc(&desc);

		if (!tf->copy_tex || tf->width != desc.Width || tf->height != desc.Height) {
			if (tf->copy_tex) {
				tf->copy_tex->Release();
				tf->copy_tex = nullptr;
			}

			D3D11_TEXTURE2D_DESC copy_desc = desc;
			copy_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
			copy_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			copy_desc.MiscFlags = 0;

			ID3D11Device *dev;
			d3d_tex->GetDevice(&dev);
			HRESULT hr = dev->CreateTexture2D(&copy_desc, nullptr, &tf->copy_tex);
			dev->Release();

			if (FAILED(hr)) {
				obs_log(LOG_ERROR, "[crevasse] Failed to create copy texture (hr=0x%08X)", hr);
			} else {
				tf->width = desc.Width;
				tf->height = desc.Height;
				obs_log(LOG_INFO, "[crevasse] Created copy texture: %ux%u", tf->width, tf->height);
			}
		}

		if (tf->copy_tex) {
			ID3D11Device *dev;
			d3d_tex->GetDevice(&dev);
			ID3D11DeviceContext *ctx;
			dev->GetImmediateContext(&ctx);
			ctx->CopyResource(tf->copy_tex, d3d_tex);
			ctx->Release();
			dev->Release();

			if (!inited_d3d11_cuda || tex_changed) {
				if (!register_d3d11_texture(tf->copy_tex, g_interop)) {
					const cudaError_t last = cudaGetLastError();
					obs_log(LOG_ERROR,
						"[crevasse] Failed to register D3D11 texture for CUDA interop (tex=%p, cuda=%d: %s)",
						tf->copy_tex, static_cast<int>(last), cudaGetErrorString(last));
				} else {
					inited_d3d11_cuda = true;
					obs_log(LOG_INFO, "[crevasse] D3D11-CUDA interop registered (tex=%p)",
						tf->copy_tex);
				}
			}
		}
	}

	static int cnt = 0;
	if (++cnt % 120 == 0) {
		blog(LOG_INFO, "[crevasse] got D3D11 texture: %p", d3d_tex);
	}

	gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	const uint32_t cx = obs_source_get_base_width(target);
	const uint32_t cy = obs_source_get_base_height(target);
	obs_source_process_filter_end(tf->source, default_effect, cx, cy);
}

struct obs_source_info crevasse_ai;

bool obs_module_load(void)
{
	crevasse_ai.id = "crevasse_ai";
	crevasse_ai.type = OBS_SOURCE_TYPE_FILTER;
	crevasse_ai.output_flags = OBS_SOURCE_VIDEO;
	crevasse_ai.get_name = filter_getname;
	crevasse_ai.create = filter_create;
	crevasse_ai.destroy = filter_destroy;
	crevasse_ai.video_render = filter_render;

	obs_register_source(&crevasse_ai);
	obs_log(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
	return true;
}

void obs_module_unload(void)
{
	obs_log(LOG_INFO, "plugin unloaded");
}