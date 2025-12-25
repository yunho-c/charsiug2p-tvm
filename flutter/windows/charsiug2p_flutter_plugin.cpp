#include "charsiug2p_flutter_plugin.h"

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>
#include <windows.h>

#include <memory>
#include <string>

namespace charsiug2p_flutter {

namespace {

std::string Utf8FromWide(const std::wstring& wide) {
  if (wide.empty()) {
    return std::string();
  }
  int size_needed = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(),
                                        static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
  if (size_needed <= 0) {
    return std::string();
  }
  std::string result(size_needed, 0);
  WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()),
                      result.data(), size_needed, nullptr, nullptr);
  return result;
}

std::string GetExecutableDir() {
  wchar_t path[MAX_PATH];
  DWORD length = GetModuleFileNameW(nullptr, path, MAX_PATH);
  if (length == 0 || length == MAX_PATH) {
    return std::string();
  }
  std::wstring full_path(path, length);
  size_t pos = full_path.find_last_of(L"\\/");
  if (pos == std::wstring::npos) {
    return std::string();
  }
  return Utf8FromWide(full_path.substr(0, pos));
}

}  // namespace

void CharsiuG2pFlutterPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows* registrar) {
  auto channel = std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
      registrar->messenger(), "charsiug2p_flutter/paths",
      &flutter::StandardMethodCodec::GetInstance());

  auto plugin = std::make_unique<CharsiuG2pFlutterPlugin>();

  channel->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto& call, auto result) {
        plugin_pointer->HandleMethodCall(call, std::move(result));
      });

  registrar->AddPlugin(std::move(plugin));
}

CharsiuG2pFlutterPlugin::CharsiuG2pFlutterPlugin() {}

CharsiuG2pFlutterPlugin::~CharsiuG2pFlutterPlugin() {}

void CharsiuG2pFlutterPlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue>& method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  if (method_call.method_name().compare("get_paths") == 0) {
    const std::string exe_dir = GetExecutableDir();
    flutter::EncodableMap paths;
    if (!exe_dir.empty()) {
      const std::string resource_dir = exe_dir + "\\data\\flutter_assets";
      paths[flutter::EncodableValue("resourceDir")] = flutter::EncodableValue(resource_dir);
      paths[flutter::EncodableValue("nativeLibraryDir")] = flutter::EncodableValue(exe_dir);
    } else {
      paths[flutter::EncodableValue("resourceDir")] = flutter::EncodableValue();
      paths[flutter::EncodableValue("nativeLibraryDir")] = flutter::EncodableValue();
    }
    result->Success(flutter::EncodableValue(paths));
    return;
  }

  result->NotImplemented();
}

}  // namespace charsiug2p_flutter
