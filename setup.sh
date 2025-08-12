#!/bin/bash

# Code Review Script Setup
echo "Setting up Code Review Script..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create example environment file
cat > .env.example << 'EOF'
# Qwen3 API Configuration
QWEN_API_BASE=http://localhost:8000/v1
QWEN_API_KEY=your-api-key-here
QWEN_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct

# Optional: Override default settings
# QWEN_MAX_TOKENS=4000
# QWEN_TEMPERATURE=0.1
EOF

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure your API settings"
echo "2. Make sure your vLLM server is running with Qwen3 model"
echo "3. Run: source venv/bin/activate"
echo "4. Run: python code_review.py --help"
