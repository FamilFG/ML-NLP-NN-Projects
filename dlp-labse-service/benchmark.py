import time
import statistics
import requests

SERVER_URL = "http://localhost:8082/embed"

TEST_QUERIES = [
    "uşaqlar üçün pambıq corablar",
    "водка пшеничная 40 градусов",
    "natural flower honey",
    "avtomobil təkərləri",
    "şokolad və kakao məmulatları",
    "günəbaxan yağı rafinə edilmiş",
    "стальные бесшовные трубы",
    "smartfonlar və mobil rabitə aparatları",
    "polad çubuqlar və armaturlar",
    "кофе жареный в зернах",
]

def run_benchmark(num_requests=30):
    print(f"🚀 Starting CPU benchmark on {SERVER_URL} ({num_requests} requests)...")
    latencies = []
    
    # Warm-up check
    try:
        requests.post(SERVER_URL, json={"inputs": "warmup"}, timeout=10)
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return

    t_start_total = time.perf_counter()
    for i in range(num_requests):
        q = TEST_QUERIES[i % len(TEST_QUERIES)]
        t0 = time.perf_counter()
        resp = requests.post(SERVER_URL, json={"inputs": q}, timeout=10)
        dt = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            latencies.append(dt)
            print(f"[{i+1:02d}/{num_requests}] {dt:6.1f} ms | '{q[:30]}...'")
        else:
            print(f"[{i+1:02d}/{num_requests}] ERR {resp.status_code}")

    total_time_s = time.perf_counter() - t_start_total
    
    if not latencies:
        print("No successful requests.")
        return

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    mean_lat = statistics.mean(latencies)
    rps = len(latencies) / total_time_s

    print("\n" + "="*50)
    print("📊 BENCHMARK RESULTS (CPU INFERENCE)")
    print("="*50)
    print(f"Total Requests:     {len(latencies)}")
    print(f"Total Duration:     {total_time_s:.2f} s")
    print(f"Throughput (RPS):   {rps:.2f} req/s")
    print(f"Mean Latency:       {mean_lat:.1f} ms")
    print(f"Median (P50):       {p50:.1f} ms")
    print(f"95th Pctl (P95):    {p95:.1f} ms")
    print(f"Min / Max Latency:  {min(latencies):.1f} ms / {max(latencies):.1f} ms")
    print("="*50)

if __name__ == "__main__":
    run_benchmark(num_requests=30)
