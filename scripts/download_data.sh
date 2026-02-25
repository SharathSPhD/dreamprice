#!/bin/bash
# Download Dominick's Finer Foods scanner data from the
# James M. Kilts Center for Marketing at University of Chicago.
#
# Usage: bash scripts/download_data.sh
#
# Note: The data requires agreement to the Kilts Center terms of use.
# Visit: https://www.chicagobooth.edu/research/kilts/datasets/dominicks

set -e
DATA_DIR="docs/data"
mkdir -p "$DATA_DIR/cso"

echo "Dominick's data must be downloaded manually from:"
echo "  https://www.chicagobooth.edu/research/kilts/datasets/dominicks"
echo ""
echo "After downloading, place files as follows:"
echo "  $DATA_DIR/cso/wcso.csv   - Canned soup movement file"
echo "  $DATA_DIR/cso/upccso.csv - Canned soup UPC lookup"
echo "  $DATA_DIR/demo.csv       - Store demographics"
echo ""
echo "Alternatively, use the preprocessed HuggingFace dataset:"
echo "  pip install datasets"
echo "  python -c \"from datasets import load_dataset; ds = load_dataset('qbz506/dreamprice-dominicks-cso')\""
