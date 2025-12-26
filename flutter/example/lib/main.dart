import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:charsiug2p_flutter/charsiug2p_flutter.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await CharsiuG2p.init();
  runApp(const G2pApp());
}

class G2pApp extends StatelessWidget {
  const G2pApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CharsiuG2P',
      home: const G2pHome(),
      theme: ThemeData.light(),
    );
  }
}

class G2pHome extends StatefulWidget {
  const G2pHome({super.key});

  @override
  State<G2pHome> createState() => _G2pHomeState();
}

class _G2pHomeState extends State<G2pHome> {
  final _assetRootController = TextEditingController();
  final _assetPrefixController =
      TextEditingController(text: 'assets/charsiug2p');
  final _langController = TextEditingController(text: 'eng-us');
  final _wordsController = TextEditingController(text: 'Char siu');

  final List<String> _targets = const [
    'metal-ios',
    'metal-macos',
    'llvm',
    'vulkan',
    'opencl',
    'cuda',
    'rocm',
    'webgpu',
  ];
  final List<String> _devices = const [
    'metal',
    'cpu',
    'vulkan',
    'opencl',
    'cuda',
    'rocm',
    'webgpu',
  ];

  CharsiuG2p? _model;
  String? _modelAssetRoot;
  bool _useBundledAssets = true;
  bool _useSystemLib = false;
  String _target = 'metal-ios';
  String _device = 'metal';
  String _result = '';
  String _error = '';
  bool _loading = false;

  @override
  void dispose() {
    _assetRootController.dispose();
    _assetPrefixController.dispose();
    _langController.dispose();
    _wordsController.dispose();
    super.dispose();
  }

  Future<void> _runG2p() async {
    setState(() {
      _loading = true;
      _error = '';
      _result = '';
    });

    try {
      String assetRoot;
      if (_useBundledAssets) {
        final prefix = _assetPrefixController.text.trim();
        if (prefix.isEmpty) {
          throw ArgumentError('Asset prefix is required.');
        }
        assetRoot =
            await CharsiuG2pAssets.prepareTokenizerRoot(assetPrefix: prefix);
        _assetRootController.text = assetRoot;
      } else {
        assetRoot = _assetRootController.text.trim();
        if (assetRoot.isEmpty) {
          throw ArgumentError('Asset root is required.');
        }
      }
      final lang = _langController.text.trim();
      if (lang.isEmpty) {
        throw ArgumentError('Language code is required.');
      }
      final words = _parseWords(_wordsController.text);
      if (words.isEmpty) {
        throw ArgumentError('Provide at least one word.');
      }

      if (_model == null || _modelAssetRoot != assetRoot) {
        _model = await CharsiuG2p.load(
          assetRoot: assetRoot,
          target: _target,
          device: _device,
          useSystemLib: _useSystemLib,
          systemLibPrefix: _useSystemLib ? 'g2p_' : null,
        );
        _modelAssetRoot = assetRoot;
      }

      final phones = await _model!.run(words, lang);
      setState(() {
        _result = phones.join('\n');
      });
    } catch (err) {
      setState(() {
        if (err is G2pFfiError) {
          _error = '${err.kind.name}: ${err.message}';
          if (err.details != null && err.details!.isNotEmpty) {
            _error = '$_error\n\nDetails:\n${err.details}';
          }
        } else {
          _error = err.toString();
        }
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  List<String> _parseWords(String raw) {
    return raw
        .split(RegExp(r'[\s,]+'))
        .map((part) => part.trim())
        .where((part) => part.isNotEmpty)
        .toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('CharsiuG2P Example')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _assetRootController,
            decoration: const InputDecoration(
              labelText: 'Asset root path',
              hintText: '/path/to/assets',
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Place assets under <assetRoot>/tokenizers/<checkpoint>/in{max_input_bytes}_out{max_output_len}/ '
            'and <assetRoot>/tvm/<checkpoint>/b{batch_size}_in{max_input_bytes}_out{max_output_len}/<target>/',
          ),
          const SizedBox(height: 12),
          SwitchListTile(
            title: const Text('Use bundled assets'),
            value: _useBundledAssets,
            onChanged: _loading
                ? null
                : (value) {
                    setState(() {
                      _useBundledAssets = value;
                    });
                  },
          ),
          if (_useBundledAssets) ...[
            const SizedBox(height: 12),
            TextField(
              controller: _assetPrefixController,
              decoration: const InputDecoration(
                labelText: 'Asset prefix',
                hintText: 'assets/charsiug2p',
              ),
            ),
          ],
          const SizedBox(height: 12),
          CheckboxListTile(
            title: const Text('Use System Lib (Static Linking)'),
            subtitle: const Text('Required for iOS'),
            value: _useSystemLib,
            onChanged: _loading
                ? null
                : (value) {
                    setState(() {
                      _useSystemLib = value ?? false;
                      // Auto-select metal-ios if system lib is enabled, as meaningful default
                      if (_useSystemLib && _target != 'metal-ios') {
                        _target = 'metal-ios';
                        _device = 'metal';
                      }
                    });
                  },
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _target,
            items: _targets
                .map((value) => DropdownMenuItem(
                      value: value,
                      child: Text(value),
                    ))
                .toList(),
            onChanged: _loading
                ? null
                : (value) {
                    if (value == null) {
                      return;
                    }
                    setState(() {
                      _target = value;
                    });
                  },
            decoration: const InputDecoration(labelText: 'Target'),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _device,
            items: _devices
                .map((value) => DropdownMenuItem(
                      value: value,
                      child: Text(value),
                    ))
                .toList(),
            onChanged: _loading
                ? null
                : (value) {
                    if (value == null) {
                      return;
                    }
                    setState(() {
                      _device = value;
                    });
                  },
            decoration: const InputDecoration(labelText: 'Device'),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _langController,
            decoration: const InputDecoration(
              labelText: 'Language code',
              hintText: 'eng-us',
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _wordsController,
            decoration: const InputDecoration(
              labelText: 'Words (space or comma separated)',
            ),
            maxLines: 2,
          ),
          const SizedBox(height: 16),
          FilledButton(
            onPressed: _loading ? null : _runG2p,
            child: Text(_loading ? 'Running...' : 'Run G2P'),
          ),
          const SizedBox(height: 16),
          if (_error.isNotEmpty) ...[
            Text('Error:', style: Theme.of(context).textTheme.titleMedium),
            SelectableText(_error),
            const SizedBox(height: 8),
            TextButton.icon(
              onPressed: () {
                Clipboard.setData(ClipboardData(text: _error));
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Error copied to clipboard')),
                );
              },
              icon: const Icon(Icons.copy),
              label: const Text('Copy Error'),
            ),
            const SizedBox(height: 16),
          ],
          if (_result.isNotEmpty) ...[
            Text('Output:', style: Theme.of(context).textTheme.titleMedium),
            Text(_result),
          ],
          const SizedBox(height: 16),
          TextButton.icon(
            onPressed: _debugAssets,
            icon: const Icon(Icons.bug_report),
            label: const Text('Debug Assets (List all)'),
          ),
        ],
      ),
    );
  }

  Future<void> _debugAssets() async {
    try {
      final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
      final assets = manifest.listAssets().toList()..sort();
      if (!mounted) return;
      showDialog(
        context: context,
        builder: (context) {
          return AlertDialog(
            title: const Text('AssetManifest'),
            content: SizedBox(
              width: double.maxFinite,
              child: ListView.builder(
                itemCount: assets.length,
                itemBuilder: (context, index) {
                  return SelectableText(assets[index]);
                },
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Close'),
              ),
            ],
          );
        },
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load manifest: $e')),
      );
    }
  }
}
