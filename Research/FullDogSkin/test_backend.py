#!/usr/bin/env python3
import sys
import os

print("="*60)
print("PYTHON BACKEND COMPREHENSIVE TEST")
print("="*60)

# Test 1: Core imports
print("\n✓ Test 1: Core module imports")
try:
    from config import *
    from model import build_model
    from data_utils import get_data_loaders
    from train_utils import train_local, test_model as test_model_fn
    from federated_utils import get_params, set_params, average_params
    from loadJson import load_disease_data
    print("  All core modules imported successfully")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 2: Flask API syntax
print("\n✓ Test 2: Flask API files syntax")
import ast
for file in ['api.py', 'FinalApi_.py', 'FinalApi_llm.py']:
    try:
        with open(file, 'r') as f:
            ast.parse(f.read())
        print(f"  {file}: Valid syntax")
    except SyntaxError as e:
        print(f"  {file}: SYNTAX ERROR - {e}")
        sys.exit(1)

# Test 3: PyTorch 
print("\n✓ Test 3: PyTorch setup")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  Device: {DEVICE}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 4: Dataset directory check
print("\n✓ Test 4: Dataset structure")
try:
    for split in ['train', 'valid', 'test']:
        path = os.path.join(DATASET_DIR, split)
        if os.path.exists(path):
            classes = os.listdir(path)
            print(f"  {split}: {len(classes)} classes found")
        else:
            print(f"  {split}: NOT FOUND (will need to be populated)")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 5: Model loading
print("\n✓ Test 5: Model building")
try:
    from model import build_model
    test_model_obj = build_model(7)  # 7 classes as per dataset
    print(f"  Model created successfully: {type(test_model_obj).__name__}")
    print(f"  Model has {sum(p.numel() for p in test_model_obj.parameters()):,} parameters")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL PYTHON TESTS PASSED - Backend ready to run!")
print("="*60)
