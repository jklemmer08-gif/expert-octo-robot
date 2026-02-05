#!/bin/bash

echo "========================================================================"
echo "TorchCodec Diagnostic Report"
echo "========================================================================"
echo ""

echo "[1] PyTorch Environment:"
python3 << 'PYCHECK'
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA version: {torch.version.cuda}")
PYCHECK

echo ""
echo "[2] Checking TorchCodec Installation:"
pip show torchcodec 2>/dev/null || echo "  (not installed yet)"

echo ""
echo "[3] Attempting TorchCodec Import:"
python3 << 'TCHECK'
import sys
try:
    import torchcodec
    print(f"  ✓ TorchCodec imported successfully: {torchcodec.__version__}")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
except Exception as e:
    print(f"  ✗ Other error: {e}")
TCHECK

echo ""
echo "[4] Checking pip list for codec-related packages:"
pip list | grep -i codec || echo "  (no codec packages found)"

echo ""
echo "[5] Installing torchcodec..."
pip install torchcodec -v 2>&1 | tail -20

echo ""
echo "[6] Retrying import after install:"
python3 << 'TRETEST'
try:
    import torchcodec
    print(f"  ✓ TorchCodec now imports successfully: {torchcodec.__version__}")
except Exception as e:
    print(f"  ✗ Still failing: {e}")
TRETEST

echo ""
echo "========================================================================"
echo "Diagnostic complete. Share the output above for analysis."
echo "========================================================================"
