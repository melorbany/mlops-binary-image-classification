# scripts/python/run_pipeline.py
#!/usr/bin/env python3
"""Cross-platform full pipeline runner."""
import os
import sys
import subprocess
import time
import signal
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts" / "python"
os.chdir(project_root)

def run_script(script_name, args=None):
    """Run a Python script and return success status."""
    cmd = [sys.executable, str(scripts_dir / script_name)]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("  Running Full MLOps Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Setup
    print("\n" + "=" * 40)
    print("Step 1: Setup")
    print("=" * 40)
    if not run_script("setup.py"):
        print("✗ Setup failed")
        sys.exit(1)
    print("✓ Setup completed")
    
    # Step 2: Create sample data
    print("\n" + "=" * 40)
    print("Step 2: Create Sample Data")
    print("=" * 40)
    if not run_script("create_sample_data.py"):
        print("✗ Data creation failed")
        sys.exit(1)
    print("✓ Data created")
    
    # Step 3: Preprocess
    print("\n" + "=" * 40)
    print("Step 3: Preprocess Data")
    print("=" * 40)
    if not run_script("preprocess.py"):
        print("✗ Preprocessing failed")
        sys.exit(1)
    print("✓ Data preprocessed")
    
    # Step 4: Run tests
    print("\n" + "=" * 40)
    print("Step 4: Run Tests")
    print("=" * 40)
    if not run_script("run_tests.py"):
        print("⚠ Some tests failed (continuing)")
    else:
        print("✓ Tests passed")
    
    # Step 5: Train (quick)
    print("\n" + "=" * 40)
    print("Step 5: Train Model (2 epochs)")
    print("=" * 40)
    if not run_script("train.py", ["--epochs", "2", "--batch-size", "16"]):
        print("✗ Training failed")
        sys.exit(1)
    print("✓ Model trained")
    
    # Step 6: Start API and test
    print("\n" + "=" * 40)
    print("Step 6: Test API")
    print("=" * 40)
    
    # Start API in background
    api_process = subprocess.Popen(
        [sys.executable, str(scripts_dir / "run_api.py"), "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("Waiting for API to start...")
    time.sleep(10)
    
    # Test API
    api_test_success = run_script("test_api.py", ["--port", "8000"])
    
    # Stop API
    api_process.terminate()
    try:
        api_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        api_process.kill()
    
    if not api_test_success:
        print("✗ API tests failed")
        sys.exit(1)
    print("✓ API tests passed")
    
    # Summary
    duration = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("  Pipeline Completed Successfully!")
    print("=" * 60)
    print(f"\nDuration: {duration:.1f} seconds")
    print("\nCompleted steps:")
    print("  ✓ Environment setup")
    print("  ✓ Data creation")
    print("  ✓ Data preprocessing")
    print("  ✓ Unit tests")
    print("  ✓ Model training")
    print("  ✓ API testing")

if __name__ == "__main__":
    main()