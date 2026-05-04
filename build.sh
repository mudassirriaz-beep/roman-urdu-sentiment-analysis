#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading model from Google Drive..."
# Install gdown
pip install gdown

# Google Drive file ID (aap ko file ka direct ID chahiye)
# Aap ka folder link hai, isme file ID nahi. Pehle file ko share karein.
# Abhi ke liye, model download skip karo (baad mein add karenge)
# gdown https://drive.google.com/uc?id=YOUR_FILE_ID -O final_model_v6.zip
# unzip -o final_model_v6.zip

echo "Build complete."