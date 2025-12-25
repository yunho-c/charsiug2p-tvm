import Cocoa
import FlutterMacOS

public class CharsiuG2pFlutterPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "charsiug2p_flutter/paths", binaryMessenger: registrar.messenger)
    let instance = CharsiuG2pFlutterPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "get_paths":
      let resourceDir: Any? = Bundle.main.resourcePath
      let nativeLibraryDir: Any? = Bundle.main.privateFrameworksPath
      result([
        "resourceDir": resourceDir ?? NSNull(),
        "nativeLibraryDir": nativeLibraryDir ?? NSNull(),
      ])
    default:
      result(FlutterMethodNotImplemented)
    }
  }
}
