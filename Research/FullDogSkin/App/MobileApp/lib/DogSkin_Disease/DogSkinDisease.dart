// ignore_for_file: file_names

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

class DogSkinDiseasePredictorPage extends StatefulWidget {
  const DogSkinDiseasePredictorPage({super.key});

  @override
  State<DogSkinDiseasePredictorPage> createState() =>
      _DogSkinDiseasePredictorPageState();
}

class _DogSkinDiseasePredictorPageState
    extends State<DogSkinDiseasePredictorPage> {
  static const String _apiBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://10.0.2.2:5001',
  );

  File? _selectedImage;
  bool _isLoading = false;
  String _errorText = '';

  String _predictedDisease = '';
  String _confidence = '';
  String _description = '';
  String _suggestion = '';
  String _whenToSeeVet = '';
  String _explanationImageBase64 = '';
  bool _isReliableResult = false;

  List<dynamic> _symptoms = [];
  List<dynamic> _causes = [];
  List<dynamic> _treatment = [];
  List<Map<String, dynamic>> _topPredictions = [];

  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(
      source: source,
      imageQuality: 85,
      maxWidth: 1200,
    );

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _errorText = '';
        _clearResultState();
      });
    }
  }

  void _showImageSourceSelector() {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(25)),
      ),
      builder: (context) => Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Select Image Source',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            ListTile(
              leading: const Icon(Icons.camera_alt, color: Colors.blue),
              title: const Text('Take Photo'),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.camera);
              },
            ),
            ListTile(
              leading: const Icon(Icons.photo, color: Colors.green),
              title: const Text('Choose from Gallery'),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.gallery);
              },
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _makePredictionRequest() async {
    if (_selectedImage == null) {
      setState(() => _errorText = 'Please select an image first.');
      return;
    }

    setState(() {
      _isLoading = true;
      _errorText = '';
    });

    try {
      final uri =
          Uri.parse('${_apiBaseUrl.replaceAll(RegExp(r"/$"), "")}/predict');
      final request = http.MultipartRequest('POST', uri)
        ..files.add(
          await http.MultipartFile.fromPath('image', _selectedImage!.path),
        );

      final response = await request.send().timeout(
            const Duration(seconds: 60),
            onTimeout: () => throw TimeoutException(
              'The server is taking too long to respond.',
            ),
          );
      final responseData = await response.stream.bytesToString();
      final data = _decodeApiResponse(responseData);

      if (response.statusCode == 200 && data['status'] == 'success') {
        _applySuccessResponse(data);
        _showResultModal();
      } else {
        _applyProblemResponse(data, response.statusCode);
        _showResultModal();
      }
    } on TimeoutException catch (e) {
      setState(() => _errorText = e.message ?? 'Request timed out.');
    } on SocketException {
      setState(() {
        _errorText =
            'Could not reach the API at $_apiBaseUrl. Check that api.py is running on port 5001.';
      });
    } catch (e) {
      debugPrint('Prediction request failed: $e');
      setState(() => _errorText = 'Connection error. Please try again.');
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Map<String, dynamic> _decodeApiResponse(String responseData) {
    if (responseData.trim().isEmpty) {
      return {'status': 'server_error', 'error': 'Empty response from server.'};
    }

    final decoded = json.decode(responseData);
    if (decoded is Map<String, dynamic>) {
      return decoded;
    }

    return {
      'status': 'server_error',
      'error': 'Unexpected response format from server.',
    };
  }

  void _applySuccessResponse(Map<String, dynamic> data) {
    setState(() {
      _isReliableResult = true;
      _predictedDisease =
          (data['predicted_disease'] ?? data['prediction'] ?? 'Unknown')
              .toString();
      _confidence = _formatPercent(data['confidence']);
      _description = data['description']?.toString() ?? '';
      _suggestion = data['warning']?.toString() ?? '';
      _symptoms = _parseList(data['symptoms']);
      _causes = _parseList(data['causes']);
      _treatment = _parseList(data['treatment']);
      _whenToSeeVet = data['when_to_see_vet']?.toString() ?? '';
      _topPredictions = _parseTopPredictions(data['top_predictions']);
      _explanationImageBase64 = _parseExplanationImage(data['explanation']);
    });
  }

  void _applyProblemResponse(Map<String, dynamic> data, int statusCode) {
    final status = data['status']?.toString() ?? 'error';
    final error = data['error']?.toString() ??
        'Prediction failed with status $statusCode.';
    final suggestion = data['suggestion']?.toString() ??
        'Try a clearer close-up photo in good lighting.';

    setState(() {
      _isReliableResult = false;
      _predictedDisease = _problemTitle(status);
      _confidence =
          _formatPercent(data['confidence'] ?? data['none_confidence']);
      _description = error;
      _suggestion = suggestion;
      _symptoms = [];
      _causes = [];
      _treatment = [];
      _whenToSeeVet = status == 'invalid_content'
          ? ''
          : 'A veterinarian should check severe, spreading, painful, or persistent skin symptoms.';
      _topPredictions = _parseTopPredictions(data['top_predictions']);
      _explanationImageBase64 = '';
    });
  }

  String _problemTitle(String status) {
    switch (status) {
      case 'low_confidence':
        return 'Needs a clearer image';
      case 'invalid_content':
        return 'Unsupported image';
      case 'service_unavailable':
        return 'Prediction service unavailable';
      case 'file_too_large':
        return 'Image too large';
      case 'bad_request':
        return 'Image could not be processed';
      default:
        return 'Prediction unavailable';
    }
  }

  String _formatPercent(dynamic value) {
    if (value == null) return '';
    final numericValue = value is num ? value : num.tryParse(value.toString());
    if (numericValue == null) return value.toString();
    return '${numericValue.toStringAsFixed(2)}%';
  }

  List<dynamic> _parseList(dynamic field) {
    if (field == null) return [];
    if (field is List) {
      return field.where((item) => item.toString().trim().isNotEmpty).toList();
    }
    if (field is String) {
      return field
          .split('\n')
          .map((item) => item.trim())
          .where((item) => item.isNotEmpty)
          .toList();
    }
    return [field.toString()];
  }

  List<Map<String, dynamic>> _parseTopPredictions(dynamic field) {
    if (field is! List) return [];
    return field
        .whereType<Map>()
        .map((item) => Map<String, dynamic>.from(item))
        .where((item) => item['label'] != null)
        .toList();
  }

  String _parseExplanationImage(dynamic explanation) {
    if (explanation is Map && explanation['image_base64'] != null) {
      return explanation['image_base64'].toString();
    }
    return '';
  }

  void _clearResultState() {
    _predictedDisease = '';
    _confidence = '';
    _description = '';
    _suggestion = '';
    _whenToSeeVet = '';
    _explanationImageBase64 = '';
    _isReliableResult = false;
    _symptoms = [];
    _causes = [];
    _treatment = [];
    _topPredictions = [];
  }

  void _showResultModal() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.92,
        decoration: BoxDecoration(
          color: Colors.grey[50],
          borderRadius: const BorderRadius.vertical(top: Radius.circular(30)),
        ),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 60,
                    height: 5,
                    decoration: BoxDecoration(
                      color: Colors.grey[300],
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                const Center(
                  child: Text(
                    'AI Dog Health Assistant',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                  ),
                ),
                const SizedBox(height: 18),
                Text(
                  _predictedDisease,
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: _isReliableResult
                        ? Colors.blueAccent
                        : Colors.orange[800],
                  ),
                ),
                const SizedBox(height: 10),
                if (_confidence.isNotEmpty) _buildConfidenceBadge(),
                const SizedBox(height: 22),
                _buildSection(
                  _isReliableResult ? Icons.description : Icons.info_outline,
                  _isReliableResult ? 'Description' : 'Result',
                  _description,
                ),
                _buildSection(
                    Icons.tips_and_updates, 'Suggestion', _suggestion),
                _buildExplanationSection(),
                _buildTopPredictionsSection(),
                _buildListSection(Icons.healing, 'Symptoms', _symptoms),
                _buildListSection(Icons.coronavirus, 'Causes', _causes),
                _buildListSection(
                    Icons.medical_services, 'Treatment', _treatment),
                if (_whenToSeeVet.isNotEmpty) _buildVetWarning(),
                const SizedBox(height: 30),
                Center(
                  child: ElevatedButton(
                    onPressed: () => Navigator.pop(context),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 40,
                        vertical: 14,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(25),
                      ),
                    ),
                    child: const Text('Close'),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildConfidenceBadge() {
    final color = _isReliableResult ? Colors.green : Colors.orange;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        _isReliableResult
            ? 'Confidence: $_confidence'
            : 'Model confidence: $_confidence',
        style: TextStyle(
          color: color.shade800,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildExplanationSection() {
    if (_explanationImageBase64.isEmpty) return const SizedBox();

    try {
      final imageBytes = base64Decode(_explanationImageBase64);
      return Container(
        margin: const EdgeInsets.only(bottom: 20),
        padding: const EdgeInsets.all(16),
        decoration: _cardDecoration(),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.visibility, color: Colors.blueAccent),
                SizedBox(width: 8),
                Text(
                  'Model Focus',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(14),
              child: Image.memory(
                imageBytes,
                width: double.infinity,
                fit: BoxFit.cover,
              ),
            ),
          ],
        ),
      );
    } catch (_) {
      return const SizedBox();
    }
  }

  Widget _buildTopPredictionsSection() {
    if (_topPredictions.isEmpty) return const SizedBox();

    return Container(
      margin: const EdgeInsets.only(bottom: 20),
      padding: const EdgeInsets.all(16),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.leaderboard, color: Colors.blueAccent),
              SizedBox(width: 8),
              Text(
                'Top Matches',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ..._topPredictions.map(_buildPredictionRow),
        ],
      ),
    );
  }

  Widget _buildPredictionRow(Map<String, dynamic> prediction) {
    final label = prediction['label']?.toString() ?? 'Unknown';
    final confidence = _formatPercent(prediction['confidence']);
    final value = prediction['confidence'];
    final numericValue = value is num ? value : num.tryParse(value.toString());
    final progress = numericValue == null
        ? 0.0
        : (numericValue.clamp(0, 100).toDouble() / 100.0);

    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  label,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
              Text(confidence),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: progress,
              minHeight: 6,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSection(IconData icon, String title, String content) {
    if (content.isEmpty) return const SizedBox();
    return Container(
      margin: const EdgeInsets.only(bottom: 20),
      padding: const EdgeInsets.all(16),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: Colors.blueAccent),
              const SizedBox(width: 8),
              Text(
                title,
                style:
                    const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(content),
        ],
      ),
    );
  }

  Widget _buildListSection(IconData icon, String title, List<dynamic> items) {
    if (items.isEmpty) return const SizedBox();
    return Container(
      margin: const EdgeInsets.only(bottom: 20),
      padding: const EdgeInsets.all(16),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: Colors.blueAccent),
              const SizedBox(width: 8),
              Text(
                title,
                style:
                    const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ...items.map(
            (item) => Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text('- $item'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildVetWarning() {
    return Container(
      margin: const EdgeInsets.only(top: 4),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.red.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(15),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.warning, color: Colors.red),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              _whenToSeeVet,
              style: TextStyle(
                color: Colors.red[800],
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }

  BoxDecoration _cardDecoration() {
    return BoxDecoration(
      color: Colors.white,
      borderRadius: BorderRadius.circular(18),
      boxShadow: const [
        BoxShadow(
          color: Colors.black12,
          blurRadius: 10,
          offset: Offset(0, 4),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 1,
        automaticallyImplyLeading: false,
        titleSpacing: 16,
        title: const Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Welcome',
              style: TextStyle(
                color: Colors.blueAccent,
                fontSize: 18,
                fontWeight: FontWeight.w600,
              ),
            ),
            IconButton(
              icon: Icon(Icons.exit_to_app, color: Colors.redAccent),
              onPressed: SystemNavigator.pop,
            ),
          ],
        ),
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                'AI Dog Health Assistant',
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Text(
                'Upload an image to detect skin disease',
                style: TextStyle(color: Colors.grey[600]),
              ),
              const SizedBox(height: 30),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(25),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(25),
                  boxShadow: const [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 20,
                      offset: Offset(0, 8),
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    GestureDetector(
                      onTap: _showImageSourceSelector,
                      child: Container(
                        height: 200,
                        width: double.infinity,
                        decoration: BoxDecoration(
                          color: Colors.grey[100],
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: Colors.blueAccent.withValues(alpha: 0.3),
                          ),
                        ),
                        child: _selectedImage != null
                            ? ClipRRect(
                                borderRadius: BorderRadius.circular(20),
                                child: Image.file(
                                  _selectedImage!,
                                  fit: BoxFit.cover,
                                ),
                              )
                            : Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const Icon(
                                    Icons.cloud_upload_rounded,
                                    size: 60,
                                    color: Colors.blueAccent,
                                  ),
                                  const SizedBox(height: 10),
                                  const Text(
                                    'No Image Selected',
                                    style:
                                        TextStyle(fontWeight: FontWeight.w500),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    'Tap to upload or capture',
                                    style: TextStyle(
                                      fontSize: 13,
                                      color: Colors.grey[600],
                                    ),
                                  ),
                                ],
                              ),
                      ),
                    ),
                    const SizedBox(height: 25),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed: () => _pickImage(ImageSource.gallery),
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Choose from Gallery'),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(15),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed: () => _pickImage(ImageSource.camera),
                        icon: const Icon(Icons.camera_alt),
                        label: const Text('Take a Photo'),
                        style: OutlinedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(15),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 25),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: _isLoading ? null : _makePredictionRequest,
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(20),
                          ),
                        ),
                        child: _isLoading
                            ? const SizedBox(
                                height: 22,
                                width: 22,
                                child: CircularProgressIndicator(
                                  color: Colors.white,
                                  strokeWidth: 2.6,
                                ),
                              )
                            : const Text(
                                'Detect Disease',
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                      ),
                    ),
                  ],
                ),
              ),
              if (_errorText.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 15),
                  child: Text(
                    _errorText,
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.red),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
