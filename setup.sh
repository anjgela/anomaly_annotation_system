#!/bin/bash
#Ottimizzato per RTX 4XXX Blackwell
echo "Setup"
#cache clean up
echo "Cache clean up..."
rm -rf /tmp/pip-* ~/.cache/pip
pip uninstall -y torch torchvision torchaudio

#PyTorch Nighlty: CUDA 12.4 (for Blackwell GPU) 2. Installazione PyTorch Nightly (CUDA 12.4)
echo "Installing PyTorch Nightly..."
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
pip install --pre torchvision torchaudio --no-deps --index-url https://download.pytorch.org/whl/nightly/cu124

#lybraries
echo "Installing requirements..."
pip install -r requirements.txt

#SAM2 + groundingDINO (frocing BlackWell architecture)
echo "Installing SAM 2 and groundingDINO..."
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
pip install --no-cache-dir --no-build-isolation sam2 rf-groundingdino

#Fix NVRTC (NVIDIA CUDA Tool)
echo "Installing CUDA NVIDIA runtime..."
pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 --quiet

#RFI AD download
echo "Cloning RFI AD repository..."
cd /workspace/annotation_system
rm -rf RFI_anomaly_detection
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

echo "Setup complete."