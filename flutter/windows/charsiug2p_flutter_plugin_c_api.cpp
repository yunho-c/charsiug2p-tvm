#include "include/charsiug2p_flutter/charsiug2p_flutter_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "charsiug2p_flutter_plugin.h"

void CharsiuG2pFlutterPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  charsiug2p_flutter::CharsiuG2pFlutterPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
