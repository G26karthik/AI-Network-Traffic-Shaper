#!/usr/bin/env python3
"""
test_smoke.py

Basic smoke tests to validate core functionality after updates.
Run this after pulling changes to ensure nothing is broken.

Usage:
    python test_smoke.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_test(name: str, passed: bool, message: str = ""):
    """Print test result with color."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} - {name}")
    if message:
        print(f"       {message}")

def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists."""
    exists = os.path.exists(path)
    print_test(f"{description} exists", exists, f"Path: {path}")
    return exists

def check_python_import(module: str) -> bool:
    """Check if a Python module can be imported."""
    try:
        __import__(module)
        print_test(f"Import {module}", True)
        return True
    except ImportError as e:
        print_test(f"Import {module}", False, f"Error: {e}")
        return False

def check_script_syntax(script_path: str) -> bool:
    """Check if a Python script has valid syntax."""
    script_name = os.path.basename(script_path)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", script_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        passed = result.returncode == 0
        msg = result.stderr if not passed else ""
        print_test(f"Syntax check: {script_name}", passed, msg)
        return passed
    except Exception as e:
        print_test(f"Syntax check: {script_name}", False, str(e))
        return False

def check_features_in_script(script_path: str) -> bool:
    """Verify dst_port is not in feature lists."""
    script_name = os.path.basename(script_path)
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'train_model.py' in script_path:
            # Check that the actual feature extraction line doesn't include dst_port
            passed = '["protocol", "length", "src_port"]' in content
            msg = "train_model.py should extract ['protocol', 'length', 'src_port'] only"
            print_test(f"Feature check: {script_name}", passed, "" if passed else msg)
            return passed
        elif 'batch_predict.py' in script_path:
            # Check that batch_predict also uses correct features
            passed = '["protocol", "length", "src_port"]' in content
            msg = "batch_predict.py should use ['protocol', 'length', 'src_port'] only"
            print_test(f"Feature check: {script_name}", passed, "" if passed else msg)
            return passed
        else:
            print_test(f"Feature check: {script_name}", True, "Skipped for this file")
            return True
    except Exception as e:
        print_test(f"Feature check: {script_name}", False, str(e))
        return False

def run_smoke_tests():
    """Run all smoke tests."""
    print(f"\n{'='*60}")
    print(f"{YELLOW}AI Traffic Shaper - Smoke Tests{RESET}")
    print(f"{'='*60}\n")
    
    results = []
    
    # Test 1: Check core dependencies
    print(f"\n{YELLOW}[1/6] Checking Core Dependencies...{RESET}")
    results.append(check_python_import("pandas"))
    results.append(check_python_import("sklearn"))
    results.append(check_python_import("joblib"))
    results.append(check_python_import("numpy"))
    results.append(check_python_import("pyshark"))
    
    # Test 2: Check optional dependencies (warnings only)
    print(f"\n{YELLOW}[2/6] Checking Optional Dependencies...{RESET}")
    torch_available = check_python_import("torch")
    if not torch_available:
        print(f"       {YELLOW}Note: PyTorch not installed (optional for deep learning){RESET}")
    
    fastapi_available = check_python_import("fastapi")
    if not fastapi_available:
        print(f"       {YELLOW}Note: FastAPI not installed (optional for API service){RESET}")
    
    # Test 3: Check core files exist
    print(f"\n{YELLOW}[3/6] Checking Core Files...{RESET}")
    results.append(check_file_exists("train_model.py", "train_model.py"))
    results.append(check_file_exists("capture_features.py", "capture_features.py"))
    results.append(check_file_exists("traffic_generator.py", "traffic_generator.py"))
    results.append(check_file_exists("batch_predict.py", "batch_predict.py"))
    results.append(check_file_exists("predict_and_shape.py", "predict_and_shape.py"))
    results.append(check_file_exists("run_pipeline.py", "run_pipeline.py"))
    
    # Test 4: Check duplicate file is removed
    print(f"\n{YELLOW}[4/6] Checking Cleanup...{RESET}")
    dup_removed = not os.path.exists("traffic_simulation/traffic_generator.py")
    print_test("Duplicate traffic_generator.py removed", dup_removed, 
               "traffic_simulation/traffic_generator.py should not exist")
    results.append(dup_removed)
    
    obsolete_files = ["label_encoder.pkl", "scaler.pkl", "network_traffic_model.pkl"]
    for file in obsolete_files:
        exists = os.path.exists(file)
        print_test(f"Obsolete file {file} removed", not exists)
        results.append(not exists)
    
    # Test 5: Check Python syntax
    print(f"\n{YELLOW}[5/6] Checking Python Syntax...{RESET}")
    scripts = [
        "train_model.py",
        "capture_features.py",
        "traffic_generator.py",
        "batch_predict.py",
        "predict_and_shape.py",
        "run_pipeline.py"
    ]
    for script in scripts:
        if os.path.exists(script):
            results.append(check_script_syntax(script))
    
    # Test 6: Check feature lists (no dst_port leakage)
    print(f"\n{YELLOW}[6/6] Checking Label Leakage Fix...{RESET}")
    results.append(check_features_in_script("train_model.py"))
    results.append(check_features_in_script("batch_predict.py"))
    
    # Summary
    print(f"\n{'='*60}")
    passed = sum(results)
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    if passed == total:
        print(f"{GREEN}✓ ALL TESTS PASSED ({passed}/{total}){RESET}")
        print(f"\n{GREEN}Project is ready to use!{RESET}")
        return 0
    else:
        print(f"{RED}✗ SOME TESTS FAILED ({passed}/{total} passed, {total-passed} failed){RESET}")
        print(f"Pass rate: {pass_rate:.1f}%")
        print(f"\n{YELLOW}Please fix the issues above before proceeding.{RESET}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = run_smoke_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Tests interrupted by user{RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{RED}Unexpected error: {e}{RESET}")
        sys.exit(1)
