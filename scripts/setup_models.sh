#!/bin/bash

# Neural Stream Fusion Engine - Model Setup Script

set -e

echo "🧠 Neural Stream Fusion Engine - Model Setup"
echo "============================================="

# Create models directory
mkdir -p models

cd models

echo "📥 Downloading Phi-3-Mini model (Q4_0 quantization)..."

# Download Phi-3-Mini model (1B parameters, Q4_0 quantization)
if [ ! -f "phi-3-mini-4k-instruct-q4.gguf" ]; then
    curl -L "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf" \
         -o "phi-3-mini-4k-instruct-q4.gguf"
    echo "✅ Phi-3-Mini model downloaded successfully"
else
    echo "✅ Phi-3-Mini model already exists"
fi

echo "📋 Model Information:"
echo "- Model: Phi-3-Mini (1B parameters)"
echo "- Quantization: Q4_0"
echo "- Context Length: 4096 tokens"
echo "- Memory Usage: ~1-2GB"

echo ""
echo "🎉 Model setup complete!"
echo "You can now start the Neural Stream Fusion Engine with:"
echo "   docker-compose up"
echo "   # or"
echo "   python main.py"
