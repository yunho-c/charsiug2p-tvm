#include "include/charsiug2p_flutter/charsiug2p_flutter_plugin.h"

#include <flutter_linux/flutter_linux.h>
#include <glib.h>
#include <string.h>

struct _CharsiuG2pFlutterPlugin {
  GObject parent_instance;
};

G_DEFINE_TYPE(CharsiuG2pFlutterPlugin, charsiug2p_flutter_plugin, g_object_get_type())

namespace {

gchar* get_executable_dir() {
  g_autofree gchar* exe_path = g_file_read_link("/proc/self/exe", nullptr);
  if (exe_path == nullptr) {
    return nullptr;
  }
  return g_path_get_dirname(exe_path);
}

FlMethodResponse* get_paths_response() {
  g_autofree gchar* exe_dir = get_executable_dir();
  FlValue* map = fl_value_new_map();

  if (exe_dir != nullptr) {
    g_autofree gchar* resource_dir = g_build_filename(exe_dir, "data", "flutter_assets", nullptr);
    g_autofree gchar* native_dir = g_build_filename(exe_dir, "lib", nullptr);
    fl_value_set_string_take(map, "resourceDir", fl_value_new_string(resource_dir));
    fl_value_set_string_take(map, "nativeLibraryDir", fl_value_new_string(native_dir));
  } else {
    fl_value_set_string_take(map, "resourceDir", fl_value_new_null());
    fl_value_set_string_take(map, "nativeLibraryDir", fl_value_new_null());
  }

  return FL_METHOD_RESPONSE(fl_method_success_response_new(map));
}

}  // namespace

static void charsiug2p_flutter_plugin_handle_method_call(
    CharsiuG2pFlutterPlugin* self,
    FlMethodCall* method_call) {
  g_autoptr(FlMethodResponse) response = nullptr;

  const gchar* method = fl_method_call_get_name(method_call);

  if (strcmp(method, "get_paths") == 0) {
    response = get_paths_response();
  } else {
    response = FL_METHOD_RESPONSE(fl_method_not_implemented_response_new());
  }

  fl_method_call_respond(method_call, response, nullptr);
}

static void charsiug2p_flutter_plugin_handle_method_call_cb(
    FlMethodChannel* channel,
    FlMethodCall* method_call,
    gpointer user_data) {
  CharsiuG2pFlutterPlugin* plugin = CHARSIUG2P_FLUTTER_PLUGIN(user_data);
  charsiug2p_flutter_plugin_handle_method_call(plugin, method_call);
}

static void charsiug2p_flutter_plugin_dispose(GObject* object) {
  G_OBJECT_CLASS(charsiug2p_flutter_plugin_parent_class)->dispose(object);
}

static void charsiug2p_flutter_plugin_class_init(CharsiuG2pFlutterPluginClass* klass) {
  G_OBJECT_CLASS(klass)->dispose = charsiug2p_flutter_plugin_dispose;
}

static void charsiug2p_flutter_plugin_init(CharsiuG2pFlutterPlugin* self) {}

void charsiug2p_flutter_plugin_register_with_registrar(FlPluginRegistrar* registrar) {
  CharsiuG2pFlutterPlugin* plugin = CHARSIUG2P_FLUTTER_PLUGIN(
      g_object_new(charsiug2p_flutter_plugin_get_type(), nullptr));

  g_autoptr(FlStandardMethodCodec) codec = fl_standard_method_codec_new();
  g_autoptr(FlMethodChannel) channel = fl_method_channel_new(
      fl_plugin_registrar_get_messenger(registrar),
      "charsiug2p_flutter/paths",
      FL_METHOD_CODEC(codec));
  fl_method_channel_set_method_call_handler(
      channel,
      charsiug2p_flutter_plugin_handle_method_call_cb,
      g_object_ref(plugin),
      g_object_unref);

  g_object_unref(plugin);
}
