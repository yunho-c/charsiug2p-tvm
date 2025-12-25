import 'package:flutter/foundation.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

ExternalLibrary? defaultExternalLibraryImpl() {
  final platform = defaultTargetPlatform;
  if (platform == TargetPlatform.iOS || platform == TargetPlatform.macOS) {
    // Rust is linked via static library in the plugin on Apple platforms.
    return ExternalLibrary.process(iKnowHowToUseIt: true);
  }
  return null;
}
