import 'package:flutter/material.dart';



// Import your Dog Skin Disease page
import 'DogSkin_Disease/dog_skin_disease.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();



  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

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