#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# --------------------------------
# Configuration
# --------------------------------
OUT_DIR="./data/things_eeg/raw_eeg"
BASE_URL="https://files.osf.io/v1/resources/crxs4/providers/googledrive"
SUBS=(01 02 03 04 05 06 07 08 09 10)

mkdir -p "$OUT_DIR"

# --------------------------------
# Step 1: Download all zip files (resume + retry)
# --------------------------------
echo "Starting dataset download..."

for sid in "${SUBS[@]}"; do
  zip_path="${OUT_DIR}/sub-${sid}.zip"
  url="${BASE_URL}/sub-${sid}.zip"

  echo "Downloading sub-${sid}.zip ..."
  wget -c --tries=10 --waitretry=5 --timeout=60 --read-timeout=60 --retry-connrefused \
       --show-progress -O "$zip_path" "$url"
done

echo "All files downloaded successfully."
echo

# --------------------------------
# Step 2: Unzip and remove only after success
# --------------------------------
echo "Extracting all zip files..."

for sid in "${SUBS[@]}"; do
  zip_path="${OUT_DIR}/sub-${sid}.zip"

  if [ -f "$zip_path" ]; then
    echo "Extracting sub-${sid}.zip ..."
    if unzip -n -q "$zip_path" -d "$OUT_DIR"; then
      echo "Removing sub-${sid}.zip"
      rm -f "$zip_path"
    else
      echo "Extraction failed for sub-${sid}.zip; keeping the file for inspection."
    fi
  else
    echo "sub-${sid}.zip already removed; skipping."
  fi
done

echo "All files extracted to: $OUT_DIR"