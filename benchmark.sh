#!/bin/bash
export VIDEO_IDX=$1
export VERBOSE=0

echo "Running clustering ..."
cd clustering
./extract_keyframes.py

cd ../

echo "Running Neural Network ..."
cd sum_gan_vae
./inference.py

cd ..

