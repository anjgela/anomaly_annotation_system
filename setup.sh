#!/bin/bash

#ffmpeg install
apt-get update && apt-get install -y ffmpeg

#optimised for modern NVIDIA GPUs
echo "Startin setup"

#PyTorch Nighlty: CUDA 12.4 (for Blackwell GPU) 2. Installazione PyTorch Nightly (CUDA 12.4)
echo "Installing PyTorch Nightly..."
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124

#lybraries
echo "Installing requirements..."
pip install -r requirements.txt

#SAM2 + groundingDINO (forcing BlackWell architecture)
echo "Installing SAM 2 and groundingDINO..."
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
pip install --no-cache-dir --no-build-isolation sam2 rf-groundingdino

#NVRTC (NVIDIA CUDA Tool)
echo "Installing CUDA NVIDIA runtime..."
pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 --quiet

#RFI AD download
echo "Cloning RFI AD repository..."
git clone https://github.com/Aleinno001/RFI_anomaly_detection.git

#patch for anomaly_detection.py and utility.py
echo "Overwriting patched files..."
cp patch_files/anomaly_detection.py RFI_anomaly_detection/anomaly_detection.py
cp patch_files/utility.py RFI_anomaly_detection/utility.py

#models download and configurations
echo "Downloading models' weights..."
mkdir -p RFI_anomaly_detection/models/grounding_dino/ RFI_anomaly_detection/models/sam2.1/ RFI_anomaly_detection/configs/grounding_dino/ realtime-detection-yolo26/

# DINO
wget -nc -q -O RFI_anomaly_detection/models/grounding_dino/groundingdino_swint_ogc.pth https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget -nc -q -O RFI_anomaly_detection/configs/grounding_dino/GroundingDINO_SwinT_OGC.py https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

# SAM 2
wget -nc -q -O RFI_anomaly_detection/models/sam2.1/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# YOLO 26
wget -nc -q -O realtime-detection-yolo26/yoloe-26l-seg.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt

# SAM 3
wget -nc -q -O sam3.pt https://huggingface.co/bodhicitta/sam3/resolve/main/sam3.pt

#patch groundingDINO
echo "Applying patch to groundingDINO..."
SED_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/groundingdino/models/GroundingDINO/transformer.py')")
sed -i 's/spatial_shapes.prod(1)/spatial_shapes.to(torch.int64).prod(1)/g' $SED_PATH

#patch: Symlink aliasing for sam3 (Pytorch NVRTC JIT)
echo "Applying NVRTC alias patch for SAM 3..."

#find Python site-packages dynamically
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

#find NVRTC files downloaded from pip (cuda12)
REAL_BUILTINS=$(find $SITE_PACKAGES/nvidia/cuda_nvrtc -name "libnvrtc-builtins.so*" | head -n 1)
REAL_NVRTC=$(find $SITE_PACKAGES/nvidia/cuda_nvrtc -name "libnvrtc.so*" | grep -v builtins | head -n 1)

if [ -n "$REAL_BUILTINS" ] && [ -n "$REAL_NVRTC" ]; then
    echo "Linking NVRTC files to simulate CUDA 13.0..."
    
    # Crea i symlink globali di sistema fingendo che siano la versione 13.0
    ln -sf $REAL_BUILTINS /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so.13.0 2>/dev/null || true
    ln -sf $REAL_NVRTC /usr/lib/x86_64-linux-gnu/libnvrtc.so.13.0 2>/dev/null || true
    ln -sf $REAL_BUILTINS /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so 2>/dev/null || true
    ln -sf $REAL_NVRTC /usr/lib/x86_64-linux-gnu/libnvrtc.so 2>/dev/null || true
    
    #update Linux linkers' cache 
    ldconfig 2>/dev/null || true
    echo "NVRTC patch applied successfully!"
else
    echo "Warning: Original NVRTC files not found, skipping patch."
fi
echo "Setup complete."