# scripts/python/run_api.py
#!/usr/bin/env python3
"""Cross-platform API runner."""
import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def main():
    parser = argparse.ArgumentParser(description='Run the API server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true', default=True)
    args = parser.parse_args()
    
    print("=" * 50)
    print("  Starting FastAPI Server")
    print("=" * 50)
    
    # Check if model exists
    model_path = project_root / "artifacts" / "model.pt"
    if not model_path.exists():
        print("\nWarning: Model not found. Creating dummy model...")
        import torch
        from src.models.cnn import SimpleCNN
        model = SimpleCNN(num_classes=2)
        model_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print("  Dummy model created.")
    
    print(f"\nServer: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()