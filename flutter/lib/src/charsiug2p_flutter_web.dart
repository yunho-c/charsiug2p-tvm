import 'dart:async';

import 'package:flutter/services.dart';
import 'package:flutter_web_plugins/flutter_web_plugins.dart';

class CharsiuG2pFlutterWeb {
  static void registerWith(Registrar registrar) {
    final channel = MethodChannel(
      'charsiug2p_flutter/paths',
      const StandardMethodCodec(),
      registrar.messenger,
    );
    channel.setMethodCallHandler(_handleMethodCall);
  }

  static Future<dynamic> _handleMethodCall(MethodCall call) async {
    switch (call.method) {
      case 'get_paths':
        return <String, dynamic>{
          'resourceDir': null,
          'nativeLibraryDir': null,
        };
      default:
        throw PlatformException(
          code: 'Unimplemented',
          message: 'charsiug2p_flutter/paths: ${call.method} not implemented on web',
        );
    }
  }
}
