#!/bin/bash
# =============================================================================
# Test Runner Script for nuwa_sdk
# =============================================================================
# This script automates the process of testing nuwa_sdk changes
# against the example project.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUWA_SDK_DIR="$(dirname "$SCRIPT_DIR")"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")/../nuwa-example/example_project"

echo "=========================================="
echo "nuwa_sdk Test Runner"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Copy updated nuwa_sdk.nim to example project"
echo "2. Build the example project"
echo "3. Run integration tests"
echo ""

# Check directories exist
if [ ! -d "$NUWA_SDK_DIR" ]; then
    echo "Error: nuwa-sdk directory not found at: $NUWA_SDK_DIR"
    exit 1
fi

if [ ! -d "$EXAMPLE_DIR" ]; then
    echo "Error: nuwa-example/example_project directory not found at: $EXAMPLE_DIR"
    exit 1
fi

echo "Step 1: Copying nuwa_sdk.nim to example project cache..."
cd "$EXAMPLE_DIR"
find .nimble/pkgs2 -name "nuwa_sdk.nim" -exec cp "$NUWA_SDK_DIR/src/nuwa_sdk.nim" {} \;
echo "✓ Copied nuwa_sdk.nim"
echo ""

echo "Step 2: Building example project..."
nuwa develop 2>&1 | tail -5
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Build successful"
else
    echo "✗ Build failed"
    exit 1
fi
echo ""

echo "Step 3: Running tests..."
pytest -v
echo ""

echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
