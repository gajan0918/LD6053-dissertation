import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';


// Import your Dog Skin Disease page
import 'DogSkin_Disease/DogSkinDisease.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();



  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Dog Skin AI',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: DogSkinDiseasePredictorPage(),
    );
  }
}