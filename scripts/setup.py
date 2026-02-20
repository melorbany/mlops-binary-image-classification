# scripts/python/setup.py
#!/usr/bin/env python3
"""Cross-platform setup script."""
import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, shell=True):
    """Run a command and return success status."""
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=False)
    return result.returncode == 0

def main():
    print("=" * 50)
    print("  MLOps Project - Cross-Platform Setup")
    print("=" * 50)
    print(f"\nSystem: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    print(f"Project root: {project_root}")
    
    # Step 1: Create virtual environment
    print("\n[Step 1] Creating virtual environment...")
    venv_path = project_root / "venv"
    if not venv_path.exists():
        run_command(f"{sys.executable} -m venv venv")
        print("  ✓ Virtual environment created")
    else:
        print("  ✓ Virtual environment already exists")
    
    # Determine pip path
    if platform.system() == "Windows":
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Step 2: Upgrade pip
    print("\n[Step 2] Upgrading pip...")
    run_command(f'"{python_path}" -m pip install --upgrade pip')
    
    # Step 3: Install dependencies
    print("\n[Step 3] Installing dependencies...")
    req_file = project_root / "requirements-dev.txt"
    if req_file.exists():
        run_command(f'"{pip_path}" install -r requirements-dev.txt')
        print("  ✓ Dependencies installed")
    else:
        print("  ! requirements-dev.txt not found")
    
    # Step 4: Create directories
    print("\n[Step 4] Creating directory structure...")
    directories = [
        "data/raw",
        "data/processed/train/cat",
        "data/processed/train/dog",
        "data/processed/val/cat",
        "data/processed/val/dog",
        "data/processed/test/cat",
        "data/processed/test/dog",
        "artifacts",
        "logs",
        "mlruns",
    ]
    for dir_path in directories:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)
    print("  ✓ Directories created")
    
    # Step 5: Create .env file
    print("\n[Step 5] Creating .env file...")
    env_file = project_root / ".env"
    if not env_file.exists():
        env_file.write_text("""LOG_LEVEL=INFO
MODEL_PATH=artifacts/model.pt
MLFLOW_TRACKING_URI=mlruns
""")
        print("  ✓ .env file created")
    else:
        print("  ✓ .env file already exists")
    
    # Step 6: Initialize DVC
    print("\n[Step 6] Initializing DVC...")
    if not (project_root / ".dvc").exists():
        run_command(f'"{python_path}" -m dvc init')
        print("  ✓ DVC initialized")
    else:
        print("  ✓ DVC already initialized")
    
    print("\n" + "=" * 50)
    print("  Setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    if platform.system() == "Windows":
        print("  1. Activate venv: .\\venv\\Scripts\\activate")
    else:
        print("  1. Activate venv: source venv/bin/activate")
    print("  2. Create data:   python scripts/create_sample_data.py")
    print("  3. Preprocess:    python scripts/preprocess.py")
    print("  4. Train model:   python scripts/train.py")
    print("  5. Run API:       python scripts/run_api.py")

if __name__ == "__main__":
    main()