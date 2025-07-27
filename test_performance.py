#!/usr/bin/env python3
"""
Performance testing script for Adobe Challenge 1a
"""

import subprocess
import time
import json
import psutil
import threading
from pathlib import Path

class PerformanceMonitor:
    def __init__(self):
        self.max_memory = 0
        self.monitoring = False
    
    def monitor_memory(self, container_name):
        """Monitor Docker container memory usage"""
        while self.monitoring:
            try:
                result = subprocess.run(
                    ["docker", "stats", container_name, "--no-stream", "--format", "table {{.MemUsage}}"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    memory_str = result.stdout.strip().split('\n')[-1]
                    if 'MiB' in memory_str or 'GiB' in memory_str:
                        # Parse memory usage
                        mem_parts = memory_str.split('/')
                        if len(mem_parts) > 0:
                            used = mem_parts[0].strip()
                            if 'GiB' in used:
                                mem_mb = float(used.replace('GiB', '')) * 1024
                            else:
                                mem_mb = float(used.replace('MiB', ''))
                            self.max_memory = max(self.max_memory, mem_mb)
            except:
                pass
            time.sleep(0.5)

def test_comprehensive_performance():
    """Comprehensive performance test"""
    print("🚀 Adobe Challenge 1a - Comprehensive Performance Test")
    print("=" * 60)
    
    # Setup
    input_dir = Path("input")
    output_dir = Path("output")
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Build image
    print("📦 Building Docker image...")
    build_start = time.time()
    build_result = subprocess.run([
        "docker", "build", "--platform", "linux/amd64", 
        "-t", "pdf-extractor-test", "."
    ], capture_output=True, text=True)
    
    build_time = time.time() - build_start
    
    if build_result.returncode != 0:
        print(f"❌ Build failed: {build_result.stderr}")
        return False
    
    print(f"✅ Build successful in {build_time:.2f}s")
    
    # Check image size
    size_result = subprocess.run([
        "docker", "images", "pdf-extractor-test", "--format", "table {{.Size}}"
    ], capture_output=True, text=True)
    
    if size_result.returncode == 0:
        size_lines = size_result.stdout.strip().split('\n')
        if len(size_lines) > 1:
            image_size = size_lines[1]
            print(f"📏 Docker image size: {image_size}")
    
    # Performance test
    print("\n🔄 Running performance test...")
    
    # Setup monitoring
    monitor = PerformanceMonitor()
    
    # Start container
    start_time = time.time()
    run_result = subprocess.Popen([
        "docker", "run", "--rm", "--name", "pdf-test-container",
        "-v", f"{Path.cwd()}/input:/app/input:ro",
        "-v", f"{Path.cwd()}/output:/app/output",
        "--network", "none",
        "pdf-extractor-test"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Start memory monitoring
    monitor.monitoring = True
    monitor_thread = threading.Thread(target=monitor.monitor_memory, args=("pdf-test-container",))
    monitor_thread.start()
    
    # Wait for completion
    stdout, stderr = run_result.communicate()
    execution_time = time.time() - start_time
    
    # Stop monitoring
    monitor.monitoring = False
    monitor_thread.join(timeout=1)
    
    print(f"⏱️  Execution time: {execution_time:.2f} seconds")
    print(f"💾 Peak memory usage: {monitor.max_memory:.1f} MB")
    
    # Validate constraints
    print("\n📋 Constraint validation:")
    time_ok = execution_time <= 10
    memory_ok = monitor.max_memory <= 16384  # 16GB in MB
    
    print(f"⏰ Time constraint (≤10s): {'✅' if time_ok else '❌'} {execution_time:.2f}s")
    print(f"💾 Memory constraint (≤16GB): {'✅' if memory_ok else '❌'} {monitor.max_memory:.1f}MB")
    
    # Check outputs
    output_files = list(output_dir.glob("*.json"))
    print(f"📄 Generated files: {len(output_files)}")
    
    total_headings = 0
    for output_file in output_files:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            headings = len(data.get('outline', []))
            total_headings += headings
            print(f"  📋 {output_file.name}: {headings} headings")
        except Exception as e:
            print(f"  ❌ {output_file.name}: Error reading - {e}")
    
    print(f"📊 Total headings extracted: {total_headings}")
    
    # Overall result
    success = (run_result.returncode == 0 and time_ok and memory_ok and len(output_files) > 0)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PERFORMANCE TEST PASSED!")
        print("✅ Ready for hackathon submission")
    else:
        print("💥 PERFORMANCE TEST FAILED!")
        print("❌ Issues need to be resolved")
    
    if stderr:
        print(f"\n📝 Container logs:\n{stderr}")
    
    return success

if __name__ == "__main__":
    success = test_comprehensive_performance()
    exit(0 if success else 1)
