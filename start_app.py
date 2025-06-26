#!/usr/bin/env python3
"""
Simple startup script for old_app that runs from parent directory
"""

import os
import sys
import subprocess

# Change to parent directory where resources are located
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent_dir)

print("✅ Starting Hybrid Product Search Engine...")
print("📊 This may take a few minutes to load the model and data...")
print("🌐 Web interface will be available at: http://127.0.0.1:8080")
print(f"📁 Working directory: {os.getcwd()}")

# Run uvicorn with the old_app
subprocess.run([
    sys.executable, "-m", "uvicorn", 
    "old_app.app:app", 
    "--host", "127.0.0.1", 
    "--port", "8080"
]) 