#!/usr/bin/env python3
"""
Local testing script for PDF Structure Extractor
"""

import subprocess
import time
import json
from pathlib import Path

def test_solution():
    """Test the solution locally"""
    print("🚀 Testing PDF Structure Extractor...")
    
    # Build Docker image
    print("📦 Building Docker image...")
    build_cmd = ["docker", "build", "--platform", "linux/amd64", "-t", "pdf-extractor", "."]
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Build failed: {result.stderr}")
        return False
    
    print("✅ Docker image built successfully")
    
    # Create test directories
    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    # Run the solution
    print("🔄 Running PDF extraction...")
    start_time = time.time()
    
    run_cmd = [
        "docker", "run", "--rm",
        "-v", f"{Path.cwd()}/input:/app/input:ro",
        "-v", f"{Path.cwd()}/output:/app/output",
        "--network", "none",
        "pdf-extractor"
    ]
    
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    execution_time = time.time() - start_time
    
    print(f"⏱️  Execution time: {execution_time:.2f} seconds")
    
    if result.returncode != 0:
        print(f"❌ Execution failed: {result.stderr}")
        return False
    
    # Check outputs
    output_files = list(Path("output").glob("*.json"))
    print(f"📄 Generated {len(output_files)} JSON files")
    
    for output_file in output_files:
        try:
            with open(output_file) as f:
                data = json.load(f)
            print(f"✅ {output_file.name}: {len(data.get('outline', []))} headings")
        except Exception as e:
            print(f"❌ Error reading {output_file.name}: {e}")
    
    return True

if __name__ == "__main__":
    success = test_solution()
    print("🎉 Test completed successfully!" if success else "💥 Test failed!")
