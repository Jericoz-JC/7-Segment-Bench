#!/bin/bash
# Automated setup for Raspberry Pi 5 (4GB)
# Run: chmod +x setup_pi.sh && ./setup_pi.sh

set -e

echo "=== 7-Segment Bench: Pi 5 Setup ==="

# System dependencies
echo "[1/6] Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    python3-opencv \
    tesseract-ocr tesseract-ocr-eng \
    libtesseract-dev \
    libatlas-base-dev libhdf5-dev \
    libgl1-mesa-glx

# Optional: Tesseract SSD trained data for 7-segment fonts
echo "[2/6] Installing Tesseract 7-segment data (if available)..."
TESSDATA_DIR="/usr/share/tesseract-ocr/5/tessdata"
if [ ! -f "$TESSDATA_DIR/ssd.traineddata" ]; then
    echo "  Note: No SSD traineddata found. Using default eng."
    echo "  For better results, add 7-segment trained data to $TESSDATA_DIR"
fi

# Python virtual environment
echo "[3/6] Creating Python virtual environment..."
cd "$(dirname "$0")"
python3 -m venv venv --system-site-packages
source venv/bin/activate

echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-pi.txt

# Create data directories
echo "[5/6] Creating data directories..."
mkdir -p data/uploads data/thumbnails instance

# Initialize database
echo "[6/6] Initializing database..."
FLASK_CONFIG=pi python3 -c "
from app import create_app
app = create_app('pi')
print('Database initialized at instance/app.db')
"

echo ""
echo "=== Setup Complete ==="
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  FLASK_CONFIG=pi python app.py"
echo ""
echo "Access at: http://$(hostname -I | awk '{print $1}'):5000"
