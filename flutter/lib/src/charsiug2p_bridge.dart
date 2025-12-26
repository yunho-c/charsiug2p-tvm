import 'api.dart';
import 'external_library.dart';
import 'frb_generated.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

class CharsiuG2p {
  final G2PModel _model;

  CharsiuG2p._(this._model);

  static Future<void> init({ExternalLibrary? externalLibrary}) async {
    final resolved = externalLibrary ?? defaultExternalLibrary();
    await G2pBridge.init(externalLibrary: resolved);
  }

  static Future<CharsiuG2p> load({
    required String assetRoot,
    String? checkpoint,
    String? target,
    int? maxInputBytes,
    int? maxOutputLen,
    int? batchSize,
    String? tvmExt,
    bool? useKvCache,
    String? device,
    int? deviceId,
    String? tokenizerRoot,
    String? tvmRoot,
    bool? useSystemLib,
    String? systemLibPrefix,
    String? postProcess,
    bool? postProcessBritish,
  }) async {
    final defaults = await g2PPlatformDefaults();
    final base = await G2pModelConfig.default_();
    final resolvedPostProcess =
        postProcess ?? base.postProcess ?? 'ipa-flap-vowel-reduced';
    final config = G2pModelConfig(
      assetRoot: assetRoot,
      checkpoint: checkpoint ?? base.checkpoint,
      target: target ?? defaults.target,
      maxInputBytes: maxInputBytes ?? base.maxInputBytes,
      maxOutputLen: maxOutputLen ?? base.maxOutputLen,
      batchSize: batchSize ?? base.batchSize,
      tvmExt: tvmExt ?? base.tvmExt,
      useKvCache: useKvCache ?? base.useKvCache,
      device: device ?? defaults.device,
      deviceId: deviceId ?? base.deviceId,
      tokenizerRoot: tokenizerRoot ?? base.tokenizerRoot,
      tvmRoot: tvmRoot ?? base.tvmRoot,
      useSystemLib: useSystemLib ?? base.useSystemLib,
      systemLibPrefix: systemLibPrefix ?? base.systemLibPrefix,
      postProcess: resolvedPostProcess,
      postProcessBritish: postProcessBritish ?? base.postProcessBritish,
    );
    final model = await g2PModelNew(config: config);
    return CharsiuG2p._(model);
  }

  Future<List<String>> run(
    List<String> words,
    String lang, {
    bool spaceAfterColon = false,
  }) {
    return g2PModelRun(
      model: _model,
      words: words,
      lang: lang,
      options: G2pRunOptions(spaceAfterColon: spaceAfterColon),
    );
  }
}
