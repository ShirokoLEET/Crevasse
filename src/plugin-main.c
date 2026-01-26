#include <obs-module.h>
#include <plugin-support.h>
#include <graphics/graphics.h>
#include <d3d11.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

struct crevasse_filter {
	obs_source_t *source;
};

static const char *filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Crevasse_ai";
}

static void filter_destroy(void *data)
{
	struct crevasse_filter *tf = data;
	if (!tf)
		return;

	bfree(tf);
}

static void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	struct crevasse_filter *tf = bzalloc(sizeof(struct crevasse_filter));

	tf->source = source;

	UNUSED_PARAMETER(settings);
	return tf;
}

static void filter_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);

	struct crevasse_filter *tf = data;
	if (!tf)
		return;

	obs_source_t *target = obs_filter_get_target(tf->source);

	if (!obs_source_process_filter_begin(tf->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING)) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	gs_texture_t *rt = gs_get_render_target();
	ID3D11Texture2D *d3d_tex = rt ? (ID3D11Texture2D *)gs_texture_get_obj(rt) : NULL;

	gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	const uint32_t cx = obs_source_get_base_width(target);
	const uint32_t cy = obs_source_get_base_height(target);
	obs_source_process_filter_end(tf->source, default_effect, cx, cy);
}

struct obs_source_info crevasse_ai = {
	.id = "crevasse_ai",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = filter_getname,
	.create = filter_create,
	.destroy = filter_destroy,
	.video_render = filter_render,
};

bool obs_module_load(void)
{
	obs_register_source(&crevasse_ai);
	obs_log(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
	return true;
}

void obs_module_unload(void)
{
	obs_log(LOG_INFO, "plugin unloaded");
}