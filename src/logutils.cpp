#include <obs-module.h>
#include <plugin-support.h>
static void log_info(const char *message)
{
	blog(LOG_INFO, "[logutils] %s", message);
}
