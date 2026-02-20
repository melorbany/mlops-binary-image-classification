# scripts/python/test_api.py
#!/usr/bin/env python3
"""Cross-platform API testing script."""
import sys
import argparse
import io

def main():
    parser = argparse.ArgumentParser(description='Test API endpoints')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 50)
    print("  Testing API Endpoints")
    print("=" * 50)
    print(f"\nTarget: {base_url}\n")
    
    try:
        import requests
    except ImportError:
        print("Error: requests not installed. Run: pip install requests")
        sys.exit(1)
    
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow not installed. Run: pip install Pillow")
        sys.exit(1)
    
    # Test 1: Health check
    print("1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ✗ Status: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Cannot connect to {base_url}")
        print("   Make sure the API server is running.")
        sys.exit(1)
    
    # Test 2: Prediction
    print("\n2. Testing /predict endpoint...")
    
    # Create test image
    img = Image.new('RGB', (224, 224), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    response = requests.post(
        f"{base_url}/predict",
        files={'file': ('test.jpg', img_bytes, 'image/jpeg')}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Status: {response.status_code}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Inference Time: {result['inference_time_ms']:.2f}ms")
    else:
        print(f"   ✗ Status: {response.status_code}")
        print(f"   Response: {response.text}")
        sys.exit(1)
    
    # Test 3: App metrics
    print("\n3. Testing /app-metrics endpoint...")
    response = requests.get(f"{base_url}/app-metrics")
    if response.status_code == 200:
        print(f"   ✓ Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    else:
        print(f"   ✗ Status: {response.status_code}")
    
    # Test 4: Prometheus metrics
    print("\n4. Testing /metrics endpoint...")
    response = requests.get(f"{base_url}/metrics")
    if response.status_code == 200:
        print(f"   ✓ Status: {response.status_code}")
        lines = response.text.split('\n')[:3]
        for line in lines:
            if line:
                print(f"   {line[:60]}...")
    else:
        print(f"   ✗ Status: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("  All API tests passed!")
    print("=" * 50)

if __name__ == "__main__":
    main()