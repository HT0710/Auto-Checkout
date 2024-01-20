#!/bin/bash

# Install rembg for background removal
pip3 install tbb rembg

# Install torch for CPU
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install ultralytics for YoloV8
pip3 install ultralytics
