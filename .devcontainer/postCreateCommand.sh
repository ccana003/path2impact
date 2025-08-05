#!/usr/bin/env bash
set -e

echo "🔹 Creating Python virtual environment..."
python3 -m venv .venv

echo "🔹 Activating virtual environment..."
source .venv/bin/activate

echo "🔹 Upgrading pip..."
pip install --upgrade pip

echo "🔹 Installing requirements..."
pip install -r requirements.txt

echo "✅ Environment ready! You can now run:"
echo "   streamlit run path2impact.py --server.port 8000 --server.address 0.0.0.0"
