#ifndef FLUTTER_PLUGIN_CHARSIUG2P_FLUTTER_PLUGIN_H_
#define FLUTTER_PLUGIN_CHARSIUG2P_FLUTTER_PLUGIN_H_

#include <flutter_linux/flutter_linux.h>

G_BEGIN_DECLS

#define CHARSIUG2P_FLUTTER_PLUGIN(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), charsiug2p_flutter_plugin_get_type(), \
                              CharsiuG2pFlutterPlugin))

G_DECLARE_FINAL_TYPE(CharsiuG2pFlutterPlugin, charsiug2p_flutter_plugin, \
                     CHARSIUG2P, FLUTTER_PLUGIN, GObject)

void charsiug2p_flutter_plugin_register_with_registrar(FlPluginRegistrar* registrar);

G_END_DECLS

#endif  // FLUTTER_PLUGIN_CHARSIUG2P_FLUTTER_PLUGIN_H_
