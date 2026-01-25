#include <obs-module.h>

struct crevasse_filter {
	obs_source_t *source;
	gs_effect_t *effect;
};

static const char *filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Crevasse - ai";
}

static void filter_destroy(void *data)
{
	struct crevasse_filter *tf = data;

	if (tf) {
		obs_enter_graphics();

		gs_effect_destroy(tf->effect);
		bfree(tf);

		obs_leave_graphics();
	}
}

static void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	struct crevasse_filter *tf = bzalloc(sizeof(struct crevasse_filter));
	char *effect_file;

	obs_enter_graphics();

	effect_file = obs_module_file("test.effect");

	tf->source = source;
	tf->effect = gs_effect_create_from_file(effect_file, NULL);
	bfree(effect_file);
	if (!tf->effect) {
		filter_destroy(tf);
		tf = NULL;
	}

	obs_leave_graphics();

	UNUSED_PARAMETER(settings);
	return tf;
}

static void filter_render(void *data, gs_effect_t *effect)
{
	struct crevasse_filter *tf = data;

	if (!obs_source_process_filter_begin(tf->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING))
		return;

	obs_source_process_filter_end(tf->source, tf->effect, 0, 0);

	UNUSED_PARAMETER(effect);
}

struct obs_source_info crevasse_filter = {
	.id = "crevasse_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = filter_getname,
	.create = filter_create,
	.destroy = filter_destroy,
	.video_render = filter_render,
};