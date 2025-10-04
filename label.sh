#!/bin/bash
# Quick launcher for the labeling UI

echo "🏷️  Starting Blog Labeling UI..."
echo ""
echo "The UI will open at: http://127.0.0.1:7860"
echo ""
echo "Press Ctrl+C to stop the server when done labeling"
echo ""

cd "$(dirname "$0")"
python src/labeling/label_ui.py
