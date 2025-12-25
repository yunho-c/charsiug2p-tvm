import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class CharsiuG2pAssets {
  const CharsiuG2pAssets._();

  /// Copy bundled assets into a writable directory and return the asset root.
  ///
  /// [assetPrefix] should match the prefix in your Flutter assets, for example:
  /// `assets/charsiug2p`.
  ///
  /// If [skipPrefixes] is provided, any asset under those subfolders is not copied.
  /// This is useful for keeping `tvm/` artifacts in the app bundle.
  static Future<String> prepareAssetRoot({
    required String assetPrefix,
    String? targetDir,
    bool overwrite = false,
    List<String> skipPrefixes = const [],
  }) async {
    final normalizedPrefix = _normalizePrefix(assetPrefix);
    final normalizedSkips = skipPrefixes
        .map((prefix) => _normalizeSubdir(prefix))
        .toList(growable: false);
    final manifest = await rootBundle.loadString('AssetManifest.json');
    final Map<String, dynamic> entries =
        jsonDecode(manifest) as Map<String, dynamic>;
    final assetKeys = entries.keys
        .where((key) => key.startsWith(normalizedPrefix))
        .where((key) => !_isSkipped(normalizedPrefix, key, normalizedSkips))
        .toList()
      ..sort();
    if (assetKeys.isEmpty) {
      throw ArgumentError('No assets found for prefix: $normalizedPrefix');
    }

    final Directory baseDir = targetDir == null
        ? await getApplicationSupportDirectory()
        : Directory(targetDir);
    await baseDir.create(recursive: true);

    for (final key in assetKeys) {
      final data = await rootBundle.load(key);
      final outPath = p.join(baseDir.path, key);
      final outFile = File(outPath);
      if (!overwrite && await outFile.exists()) {
        continue;
      }
      await outFile.parent.create(recursive: true);
      await outFile.writeAsBytes(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
        flush: true,
      );
    }

    final root = normalizedPrefix.endsWith('/')
        ? normalizedPrefix.substring(0, normalizedPrefix.length - 1)
        : normalizedPrefix;
    return p.join(baseDir.path, root);
  }

  static String _normalizePrefix(String prefix) {
    final trimmed = prefix.trim();
    if (trimmed.isEmpty) {
      throw ArgumentError('assetPrefix must be non-empty');
    }
    return trimmed.endsWith('/') ? trimmed : '$trimmed/';
  }

  static String _normalizeSubdir(String prefix) {
    final trimmed = prefix.trim().replaceAll('\\', '/');
    if (trimmed.isEmpty) {
      throw ArgumentError('skipPrefixes entries must be non-empty');
    }
    return trimmed.endsWith('/') ? trimmed : '$trimmed/';
  }

  static bool _isSkipped(
      String basePrefix, String key, List<String> skipPrefixes) {
    final relative = key.substring(basePrefix.length);
    for (final skip in skipPrefixes) {
      if (relative.startsWith(skip)) {
        return true;
      }
    }
    return false;
  }

  /// Copy only tokenizer assets and return the resulting asset root path.
  ///
  /// This skips `tvm/` so compiled artifacts remain in the app bundle.
  static Future<String> prepareTokenizerRoot({
    required String assetPrefix,
    String? targetDir,
    bool overwrite = false,
  }) async {
    return prepareAssetRoot(
      assetPrefix: assetPrefix,
      targetDir: targetDir,
      overwrite: overwrite,
      skipPrefixes: const ['tvm/'],
    );
  }
}
