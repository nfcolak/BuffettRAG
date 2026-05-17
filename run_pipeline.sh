#!/bin/bash
# Quick start script for the RAG ingestion pipeline

set -e  # Exit on error

echo "=================================="
echo "RAG Ingestion Pipeline Quick Start"
echo "=================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Create processed_data directory
mkdir -p processed_data
echo "✓ Created output directory"

# Run the pipeline
echo ""
echo "Starting ingestion pipeline..."
echo "=================================="
python3 ingestion_pipeline.py

echo ""
echo "Pipeline complete! 🎉"
echo ""
echo "Validate results with:"
echo "  python3 validate_chunks.py"
echo "  python3 validate_chunks.py --sample 5"
echo "  python3 validate_chunks.py --year 1977"
