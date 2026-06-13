import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart';
class DogSkinDiseasePredictorPage extends StatefulWidget {
  const DogSkinDiseasePredictorPage({super.key});

  @override
  State<DogSkinDiseasePredictorPage> createState() =>
      _DogSkinDiseasePredictorPageState();
}

class _DogSkinDiseasePredictorPageState
    extends State<DogSkinDiseasePredictorPage> {

  File? _selectedImage;

  String _predictedDisease = "";
  String _confidence = "";
  String _description = "";
  String _whenToSeeVet = "";

  List<dynamic> _symptoms = [];
  List<dynamic> _causes = [];
  List<dynamic> _treatment = [];

  bool _isLoading = false;
  String _errorText = "";

  final ImagePicker _picker = ImagePicker();

  // ================= IMAGE PICKER =================
  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(
      source: source,
      imageQuality: 85,
      maxWidth: 1200,
    );

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _errorText = "";
      });
    }
  }

  void _showImageSourceSelector() {
    showModalBottomSheet(
      context: context,
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(25))),
      builder: (context) => Padding(
        padding: EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text("Select Image Source",
                style: TextStyle(
                    fontSize: 18, fontWeight: FontWeight.bold)),
            SizedBox(height: 20),
            ListTile(
              leading: Icon(Icons.camera_alt, color: Colors.blue),
              title: Text("Take Photo"),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.camera);
              },
            ),
            ListTile(
              leading: Icon(Icons.photo, color: Colors.green),
              title: Text("Choose from Gallery"),
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

  // ================= API =================
  Future<void> _makePredictionRequest() async {
    if (_selectedImage == null) {
      setState(() => _errorText = "Please select an image first.");
      return;
    }

    setState(() => _isLoading = true);

    try {
      var uri = Uri.parse('http://192.168.1.196:5001/predict');

      var request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath(
            'image', _selectedImage!.path));

      // Add a longer timeout (60 seconds for OpenAI API)
      var response = await request.send().timeout(
        Duration(seconds: 60),
        onTimeout: () {
          throw Exception('Request timed out. The server is taking too long to respond.');
        },
      );

      var responseData = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final data = json.decode(responseData);

        if (data['status'] == 'success') {
          setState(() {
            _predictedDisease = data['predicted_disease'];
            _confidence = "${data['confidence']}%";
            _description = data['description'] ?? "";

            // Handle both string and list formats
            _symptoms = _parseField(data['symptoms']);
            _causes = _parseField(data['causes']);
            _treatment = _parseField(data['treatment']);
            _whenToSeeVet = data['when_to_see_vet'] ?? "";
          });

          _showResultModal();
        }
      } else if (response.statusCode == 400) {
        final data = json.decode(responseData);
        
        if (data['status'] == 'low_confidence') {
          setState(() => _errorText = "❌ Image too unclear (${data['confidence']}% confidence)\n\n${data['error']}\n\n💡 ${data['suggestion']}");
        } else if (data['status'] == 'invalid_content') {
          setState(() => _errorText = "❌ Not a dog skin image\n\n${data['error']}\n\n💡 ${data['suggestion']}");
        } else {
          setState(() => _errorText = data['error'] ?? "Invalid request: ${response.statusCode}");
        }
      } else {
        setState(() => _errorText = "Server error: ${response.statusCode}");
      }
    } catch (e) {
      print('Error: $e');
      setState(() => _errorText = "Connection error: ${e.toString()}");
    }

    setState(() => _isLoading = false);
  }

// Helper method to parse fields that could be strings or lists
  List<dynamic> _parseField(dynamic field) {
    if (field == null) return [];
    if (field is String) {
      // Split by bullet points or newlines if it's a string
      return field.split('\n').where((s) => s.trim().isNotEmpty).toList();
    }
    if (field is List) return field;
    return [];
  }

  // ================= RESULT MODAL =================
  void _showResultModal() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.92,
        decoration: BoxDecoration(
          color: Colors.grey[50],
          borderRadius:
          BorderRadius.vertical(top: Radius.circular(30)),
        ),
        child: Padding(
          padding: EdgeInsets.all(20),
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

                SizedBox(height: 20),

                Center(
                  child: Text(
                    "AI Dog Health Assistant",
                    style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold),
                  ),
                ),

                SizedBox(height: 10),

                // Disease Name
                Text(
                  _predictedDisease,
                  style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.blueAccent),
                ),

                SizedBox(height: 10),

                // Confidence Badge
                Container(
                  padding:
                  EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.green.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    "Confidence: $_confidence",
                    style: TextStyle(
                        color: Colors.green[800],
                        fontWeight: FontWeight.bold),
                  ),
                ),

                SizedBox(height: 25),

                _buildSection(
                    Icons.description, "Description", _description),

                _buildListSection(
                    Icons.healing, "Symptoms", _symptoms),

                _buildListSection(
                    Icons.coronavirus, "Causes", _causes),

                _buildListSection(
                    Icons.medical_services, "Treatment", _treatment),

                if (_whenToSeeVet.isNotEmpty)
                  Container(
                    margin: EdgeInsets.only(top: 20),
                    padding: EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.red.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.warning,
                            color: Colors.red),
                        SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            _whenToSeeVet,
                            style: TextStyle(
                                color: Colors.red[800],
                                fontWeight: FontWeight.w500),
                          ),
                        )
                      ],
                    ),
                  ),

                SizedBox(height: 30),

                Center(
                  child: ElevatedButton(
                    onPressed: () => Navigator.pop(context),
                    style: ElevatedButton.styleFrom(
                        padding: EdgeInsets.symmetric(
                            horizontal: 40, vertical: 14),
                        shape: RoundedRectangleBorder(
                            borderRadius:
                            BorderRadius.circular(25))),
                    child: Text("Close"),
                  ),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSection(IconData icon, String title, String content) {
    if (content.isEmpty) return SizedBox();
    return Container(
      margin: EdgeInsets.only(bottom: 20),
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
              color: Colors.black12,
              blurRadius: 10,
              offset: Offset(0, 4))
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: Colors.blueAccent),
              SizedBox(width: 8),
              Text(title,
                  style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold)),
            ],
          ),
          SizedBox(height: 10),
          Text(content),
        ],
      ),
    );
  }

  Widget _buildListSection(
      IconData icon, String title, List<dynamic> items) {
    if (items.isEmpty) return SizedBox();
    return Container(
      margin: EdgeInsets.only(bottom: 20),
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
              color: Colors.black12,
              blurRadius: 10,
              offset: Offset(0, 4))
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: Colors.blueAccent),
              SizedBox(width: 8),
              Text(title,
                  style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold)),
            ],
          ),
          SizedBox(height: 10),
          ...items.map((e) => Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Text("• $e"),
          ))
        ],
      ),
    );
  }

  // ================= MAIN UI =================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100],



      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 1,
        automaticallyImplyLeading: false,
        titleSpacing: 16,
        title: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [

            // LEFT - Welcome Text
            Text(
              "Welcome",
              style: TextStyle(
                color: Colors.blueAccent,
                fontSize: 18,
                fontWeight: FontWeight.w600,
              ),
            ),

            // RIGHT - Exit Icon
            IconButton(
              icon: Icon(
                Icons.exit_to_app,
                color: Colors.redAccent,
              ),
              onPressed: () {
                SystemNavigator.pop(); // Proper app exit
              },
            ),
          ],
        ),
      ),



      body: Center(
        child: SingleChildScrollView(
          padding: EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [

              // Title
              Text(
                "AI Dog Health Assistant",
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),

              SizedBox(height: 8),

              Text(
                "Upload a DOG photo to detect skin disease",
                style: TextStyle(
                  color: Colors.grey[600],
                  fontWeight: FontWeight.w500,
                ),
              ),

              SizedBox(height: 12),

              // WARNING MESSAGE
              Container(
                padding: EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.orange[50],
                  border: Border.all(color: Colors.orange[300]!, width: 1.5),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(Icons.warning_amber_rounded, color: Colors.orange[700], size: 20),
                    SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        "This app is for DOG skin diseases only. Please upload dog photos only.",
                        style: TextStyle(
                          color: Colors.orange[900],
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              SizedBox(height: 30),

              // IMAGE CARD
              Container(
                width: double.infinity,
                padding: EdgeInsets.all(25),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(25),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 20,
                      offset: Offset(0, 8),
                    )
                  ],
                ),
                child: Column(
                  children: [

                    // Image Preview Area
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
                          borderRadius:
                          BorderRadius.circular(20),
                          child: Image.file(
                            _selectedImage!,
                            fit: BoxFit.cover,
                          ),
                        )
                            : Column(
                          mainAxisAlignment:
                          MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.cloud_upload_rounded,
                              size: 60,
                              color: Colors.blueAccent,
                            ),
                            SizedBox(height: 10),
                            Text(
                              "No Image Selected",
                              style: TextStyle(
                                  fontWeight: FontWeight.w500),
                            ),
                            SizedBox(height: 4),
                            Text(
                              "Tap to upload or capture",
                              style: TextStyle(
                                  fontSize: 13,
                                  color: Colors.grey),
                            ),
                          ],
                        ),
                      ),
                    ),

                    SizedBox(height: 25),

                    // Gallery Button
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed: () =>
                            _pickImage(ImageSource.gallery),
                        icon: Icon(Icons.photo_library),
                        label: Text("Choose from Gallery"),
                        style: ElevatedButton.styleFrom(
                          padding: EdgeInsets.symmetric(
                              vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius:
                            BorderRadius.circular(15),
                          ),
                        ),
                      ),
                    ),

                    SizedBox(height: 12),

                    // Camera Button
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed: () =>
                            _pickImage(ImageSource.camera),
                        icon: Icon(Icons.camera_alt),
                        label: Text("Take a Photo"),
                        style: OutlinedButton.styleFrom(
                          padding: EdgeInsets.symmetric(
                              vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius:
                            BorderRadius.circular(15),
                          ),
                        ),
                      ),
                    ),

                    SizedBox(height: 25),

                    // Detect Button
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: _isLoading
                            ? null
                            : _makePredictionRequest,
                        style: ElevatedButton.styleFrom(
                          padding: EdgeInsets.symmetric(
                              vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius:
                            BorderRadius.circular(20),
                          ),
                        ),
                        child: _isLoading
                            ? CircularProgressIndicator(
                            color: Colors.white)
                            : Text(
                          "Detect Disease",
                          style: TextStyle(
                              fontSize: 16,
                              fontWeight:
                              FontWeight.bold),
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              if (_errorText.isNotEmpty)
                Padding(
                  padding: EdgeInsets.only(top: 15),
                  child: Container(
                    padding: EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: Colors.red[50],
                      border: Border.all(color: Colors.red[400]!, width: 1.5),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(Icons.error_outline, color: Colors.red[700], size: 22),
                            SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                "Error",
                                style: TextStyle(
                                  color: Colors.red[900],
                                  fontWeight: FontWeight.bold,
                                  fontSize: 15,
                                ),
                              ),
                            ),
                          ],
                        ),
                        SizedBox(height: 8),
                        Text(
                          _errorText,
                          style: TextStyle(
                            color: Colors.red[800],
                            fontSize: 13,
                            height: 1.5,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }


}