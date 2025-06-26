#!/bin/bash

# Go to parent directory where all resources are located
cd "$(dirname "$0")/.."

# Suppress PyTorch DataParallel warnings
export PYTHONWARNINGS="ignore::UserWarning"

# Start the old_app
echo "✅ Starting Hybrid Product Search Engine from old_app..."
echo "📊 This may take a few minutes to load the model and data..."
echo "🌐 Web interface will be available at: http://127.0.0.1:8080"

# Run the app
python -m uvicorn old_app.app:app --host 127.0.0.1 --port 8080 