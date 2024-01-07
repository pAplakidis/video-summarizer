#!/bin/bash
export VERBOSE=0

for ((i=0; i<25; i++))
do
  export VIDEO_IDX=$i

  echo "Running clustering ..."
  cd clustering
  ./extract_keyframes.py
  echo "\n"
  cd ../

  echo "Running Neural Network ..."
  cd sum_gan_vae
  ./inference.py
  echo "\n"
  cd ..
done

