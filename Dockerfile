#base image with CUDA 12.4 and C++ compilers required for PyTorch/SAM2
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

#set non-interactive mode for apt to prevent installation prompts
ENV DEBIAN_FRONTEND=noninteractive

#install system dependencies (OpenCV requires libgl1 and libglib2.0-0)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

#alias python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

#set the working directory inside the container
WORKDIR /app

#copy the entire repository into the container
COPY . /app/

#make the setup script executable and run it to install models and patches
RUN chmod +x setup.sh && ./setup.sh

#expose Streamlit's default port
EXPOSE 8501

#command to run the application using the configuration from .env
CMD ["sh", "-c", "export $(grep -v '^#' .env | xargs) && streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]