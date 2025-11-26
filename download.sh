#!/bin/bash
set -e

# =============================================================================
# LibriSpeech + RIR Setup Script for H100
# =============================================================================

DATA_DIR="${DATA_DIR:-$HOME/datasets}"
LIBRISPEECH_DIR="$DATA_DIR/LibriSpeech"
RIR_DIR="$DATA_DIR/librispeech_rirs"

echo "============================================"
echo "Setup directories:"
echo "  LibriSpeech: $LIBRISPEECH_DIR"
echo "  RIRs: $RIR_DIR"
echo "============================================"

# -----------------------------------------------------------------------------
# 1. Download LibriSpeech
# -----------------------------------------------------------------------------
echo ""
echo "[1/2] Downloading LibriSpeech..."

mkdir -p "$LIBRISPEECH_DIR"
cd "$LIBRISPEECH_DIR"

# Training data (~6GB, 100 hours)
if [ ! -d "train-clean-100" ]; then
    echo "Downloading train-clean-100..."
    wget -q --show-progress https://www.openslr.org/resources/12/train-clean-100.tar.gz
    tar -xzf train-clean-100.tar.gz
    rm train-clean-100.tar.gz
else
    echo "train-clean-100 already exists, skipping."
fi

# Validation data (~350MB)
if [ ! -d "dev-clean" ]; then
    echo "Downloading dev-clean..."
    wget -q --show-progress https://www.openslr.org/resources/12/dev-clean.tar.gz
    tar -xzf dev-clean.tar.gz
    rm dev-clean.tar.gz
else
    echo "dev-clean already exists, skipping."
fi

# Test data (~350MB)
if [ ! -d "test-clean" ]; then
    echo "Downloading test-clean..."
    wget -q --show-progress https://www.openslr.org/resources/12/test-clean.tar.gz
    tar -xzf test-clean.tar.gz
    rm test-clean.tar.gz
else
    echo "test-clean already exists, skipping."
fi

echo "LibriSpeech download complete."
echo "  train-clean-100: $(find train-clean-100 -name '*.flac' | wc -l) utterances"
echo "  dev-clean: $(find dev-clean -name '*.flac' | wc -l) utterances"
echo "  test-clean: $(find test-clean -name '*.flac' | wc -l) utterances"

