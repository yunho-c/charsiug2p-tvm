import 'package:flutter/services.dart';

class CharsiuG2pPlatformPaths {
  final String? resourceDir;
  final String? nativeLibraryDir;

  const CharsiuG2pPlatformPaths({
    required this.resourceDir,
    required this.nativeLibraryDir,
  });
}

class CharsiuG2pPlatformChannels {
  CharsiuG2pPlatformChannels._();

  static const MethodChannel _channel = MethodChannel('charsiug2p_flutter/paths');

  static Future<CharsiuG2pPlatformPaths> getPaths() async {
    final Map<dynamic, dynamic>? raw = await _channel.invokeMethod('get_paths');
    return CharsiuG2pPlatformPaths(
      resourceDir: raw?['resourceDir'] as String?,
      nativeLibraryDir: raw?['nativeLibraryDir'] as String?,
    );
  }
}
