#!/bin/bash

echo "Checking CUDA and cuDNN..."
echo

check_cuda() {
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | awk -F. '{print $1"."$2}')
    
    if [ -z "$cuda_version" ]; then
        echo "CUDA not found. Please make sure CUDA is installed."
        exit 1
    fi
    
    echo "CUDA version: $cuda_version"
}

check_cudnn() {
    cudnn_path="/usr/local/cuda/include/cudnn_version.h"

    if [ ! -e $cudnn_path ]; then
        echo "cuDNN not found. Please make sure cuDNN is installed."
        exit 1
    fi

    major=$(grep '#define CUDNN_MAJOR' $cudnn_path | awk '{print $3}')
    minor=$(grep '#define CUDNN_MINOR' $cudnn_path | awk '{print $3}')

    cudnn_version="${major}.${minor}"

    echo "cuDNN version: $cudnn_version"
}

# Check cuda
check_cuda

# Check cudnn
check_cudnn

echo --------------------

# Install rembg[gpu] for background removal
pip3 install tbb rembg[gpu]

# Reinstall onnxruntime-gpu so it can work properly
pip3 uninstall onnxruntime onnxruntime-gpu -y
pip3 install onnxruntime-gpu

# Set the PyTorch CUDA version
index_url=https://download.pytorch.org/whl/cu$(echo "$cuda_version" | awk '{gsub(/\./,""); print}')

# Install torch for GPU
pip3 install torch torchvision --index-url $index_url

# Install ultralytics for YoloV8
pip3 install ultralytics

# Others
pip3 install rich rootutils hydra-core==1.3.0
