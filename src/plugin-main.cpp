#include <obs-module.h>
#include <plugin-support.h>
#include <graphics/graphics.h>
#include <d3d11.h>
#include "d3d_cuda_interop.h"
#include "infer.h"
#include <cuda_runtime.h>
#include <vector>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

struct crevasse_filter {
	obs_source_t *source;
	gs_texrender_t *texrender = nullptr;
	uint32_t width = 0;
	uint32_t height = 0;
	infer *inference = nullptr;
	std::vector<float> host_output;
};

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

	unregister_d3d11_texture(g_interop);

	if (tf->texrender) {
		gs_texrender_destroy(tf->texrender);
		tf->texrender = nullptr;
	}

	if (tf->inference) {
		delete tf->inference;
		tf->inference = nullptr;
	}

	bfree(tf);
}

static void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	auto *tf = static_cast<crevasse_filter *>(bzalloc(sizeof(crevasse_filter)));

	tf->source = source;
	tf->inference = new infer();

	// Hardcoded model path for now - normally this would come from settings
	// Assuming the user will place the engine file here for testing
	if (!tf->inference->init("D:\\model.engine")) {
		blog(LOG_ERROR, "[crevasse] Failed to init inference engine");
	} else {
		// Output size 1x5x8400
		tf->host_output.resize(1 * 5 * 8400);
	}

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
	obs_source_t *parent = obs_filter_get_parent(tf->source);

	if (!target || !parent) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	const uint32_t cx = obs_source_get_base_width(target);
	const uint32_t cy = obs_source_get_base_height(target);

	if (cx == 0 || cy == 0) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	// Update dimensions if changed
	if (tf->width != cx || tf->height != cy) {
		tf->width = cx;
		tf->height = cy;

		// Re-create texrender
		if (tf->texrender) {
			gs_texrender_destroy(tf->texrender);
			tf->texrender = nullptr;
		}

		// Unregister old CUDA resource
		unregister_d3d11_texture(g_interop);

		blog(LOG_INFO, "[crevasse] Resize: %ux%u", cx, cy);
	}

	// Create texrender if needed
	if (!tf->texrender) {
		tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
		if (!tf->texrender) {
			obs_source_skip_video_filter(tf->source);
			return;
		}
	}

	// Step 1: Render source to our texrender
	gs_texrender_reset(tf->texrender);

	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

	if (gs_texrender_begin(tf->texrender, cx, cy)) {
		struct vec4 clear_color;
		vec4_zero(&clear_color);
		gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
		gs_ortho(0.0f, (float)cx, 0.0f, (float)cy, -100.0f, 100.0f);

		// Render the source content
		obs_source_video_render(target);

		gs_texrender_end(tf->texrender);
	}

	gs_blend_state_pop();

	// Step 2: Get the rendered texture and process with CUDA
	gs_texture_t *tex = gs_texrender_get_texture(tf->texrender);
	if (tex) {
		ID3D11Texture2D *d3d_tex = static_cast<ID3D11Texture2D *>(gs_texture_get_obj(tex));

		if (d3d_tex) {
			// Get device and context for flush
			ID3D11Device *dev;
			d3d_tex->GetDevice(&dev);
			ID3D11DeviceContext *ctx;
			dev->GetImmediateContext(&ctx);

			// Flush to ensure D3D commands are completed before CUDA access
			ctx->Flush();

			// Initialize CUDA-D3D11 interop on first use
			static bool cuda_init_attempted = false;
			if (!cuda_init_attempted) {
				cuda_init_attempted = true;
				if (!init_cuda_d3d11_device(dev)) {
					obs_log(LOG_WARNING, "[crevasse] Failed to init CUDA-D3D11 device association");
				}
			}

			// Register for CUDA interop and run inference
			if (register_d3d11_texture(d3d_tex, g_interop)) {
				if (tf->inference) {
					tf->inference->forward(g_interop, tf->host_output.data());
				}
			} else {
				const cudaError_t last = cudaGetLastError();
				obs_log(LOG_ERROR,
					"[crevasse] Failed to register D3D11 texture for CUDA interop (tex=%p, cuda=%d: %s)",
					d3d_tex, static_cast<int>(last), cudaGetErrorString(last));
			}

			ctx->Release();
			dev->Release();
		}
	}

	// Log inference results periodically
	static int cnt = 0;
	if (++cnt % 120 == 0) {
		// Model output: 1x5x8400 (Format: cx, cy, w, h, conf)
		// Layout: Planar [Channels][Anchors] -> [5][8400]
		constexpr int kNumAnchors = 8400;
		constexpr int kNumChannels = 5; // cx, cy, w, h, conf
		constexpr float kConfThreshold = 0.4f;

		if (tf->host_output.size() >= kNumAnchors * kNumChannels) {
			const float *data_ptr = tf->host_output.data();
			const float *conf_data = data_ptr + (4 * kNumAnchors);

			// Find anchor with highest confidence
			int best_i = -1;
			float best_conf = kConfThreshold;

			for (int i = 0; i < kNumAnchors; ++i) {
				if (conf_data[i] > best_conf) {
					best_conf = conf_data[i];
					best_i = i;
				}
			}

			if (best_i >= 0) {
				const float cx = data_ptr[0 * kNumAnchors + best_i];
				const float cy = data_ptr[1 * kNumAnchors + best_i];
				const float w = data_ptr[2 * kNumAnchors + best_i];
				const float h = data_ptr[3 * kNumAnchors + best_i];

				blog(LOG_INFO, "[crevasse] Best: cx=%.1f cy=%.1f w=%.1f h=%.1f conf=%.3f", cx, cy, w, h,
				     best_conf);
			} else {
				blog(LOG_INFO, "[crevasse] No detection above threshold %.2f", kConfThreshold);
			}
		}
	}

	// Step 3: Draw filter output (passthrough - draw the rendered texture)
	gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	if (obs_source_process_filter_begin(tf->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING)) {
		obs_source_process_filter_end(tf->source, default_effect, cx, cy);
	}
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