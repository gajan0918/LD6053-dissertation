import 'package:flutter_test/flutter_test.dart';

import 'package:new_app/main.dart';

void main() {
  testWidgets('Dog skin assistant screen renders', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    expect(find.text('AI Dog Health Assistant'), findsOneWidget);
    expect(find.text('Detect Disease'), findsOneWidget);
  });
}
