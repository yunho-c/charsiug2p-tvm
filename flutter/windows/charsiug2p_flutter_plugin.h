#ifndef FLUTTER_PLUGIN_CHARSIUG2P_FLUTTER_PLUGIN_H_
#define FLUTTER_PLUGIN_CHARSIUG2P_FLUTTER_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace charsiug2p_flutter {

class CharsiuG2pFlutterPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar);

  CharsiuG2pFlutterPlugin();

  virtual ~CharsiuG2pFlutterPlugin();

  CharsiuG2pFlutterPlugin(const CharsiuG2pFlutterPlugin&) = delete;
  CharsiuG2pFlutterPlugin& operator=(const CharsiuG2pFlutterPlugin&) = delete;

 private:
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace charsiug2p_flutter

#endif  // FLUTTER_PLUGIN_CHARSIUG2P_FLUTTER_PLUGIN_H_
