import time
import requests

SERVER_URL = "http://localhost:8082"

def test_model():
    print(f"1. Checking health: {SERVER_URL}/health ...")
    try:
        r_health = requests.get(f"{SERVER_URL}/health", timeout=5)
        print("   Health response:", r_health.status_code, r_health.json())
    except Exception as e:
        print(f"❌ Server not reachable at {SERVER_URL}: {e}")
        return

    test_query = "uşaqlar üçün pambıq corablar"
    print(f"\n2. Testing embedding for: '{test_query}' ...")
    
    t0 = time.perf_counter()
    resp = requests.post(
        f"{SERVER_URL}/embed",
        json={"inputs": test_query},
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    if resp.status_code == 200:
        data = resp.json()
        vector = data[0] if isinstance(data, list) else data.get("embeddings", [[]])[0]
        dim = len(vector)
        # Check L2 norm (should be ~1.0)
        norm = sum(x**2 for x in vector)**0.5
        print(f"   ✅ SUCCESS!")
        print(f"   Status Code: {resp.status_code}")
        print(f"   Vector Dimensions: {dim} (Expected: 768)")
        print(f"   L2 Norm: {norm:.4f} (Expected: ~1.0)")
        print(f"   Latency: {elapsed_ms:.2f} ms")
        print(f"   Sample values: {vector[:5]} ...")
    else:
        print(f"   ❌ FAILED with code {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    test_model()
