import 'package:flutter/material.dart';
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
  final _langController = TextEditingController(text: 'eng-us');
  final _wordsController = TextEditingController(text: 'Char siu');

  CharsiuG2p? _model;
  String? _modelAssetRoot;
  String _result = '';
  String _error = '';
  bool _loading = false;

  @override
  void dispose() {
    _assetRootController.dispose();
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
      final assetRoot = _assetRootController.text.trim();
      if (assetRoot.isEmpty) {
        throw ArgumentError('Asset root is required.');
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
        _model = await CharsiuG2p.load(assetRoot: assetRoot);
        _modelAssetRoot = assetRoot;
      }

      final phones = await _model!.run(words, lang);
      setState(() {
        _result = phones.join('\n');
      });
    } catch (err) {
      setState(() {
        _error = err.toString();
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
            Text(_error),
            const SizedBox(height: 16),
          ],
          if (_result.isNotEmpty) ...[
            Text('Output:', style: Theme.of(context).textTheme.titleMedium),
            Text(_result),
          ],
        ],
      ),
    );
  }
}
