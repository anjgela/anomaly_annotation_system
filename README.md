# ANOMALY ANNOTATION SIYSTEM IN THERMAL VIDEOS

Semi-automatic annotation system of anomalies in thermal videos


##Installation and Execution

This project provides two ways to run the system: a Docker container for immediate reproduction, or a Conda environment for local development. 
Both methods require an NVIDIA GPU (optimised for RTX 4000/Blackwell series via CUDA 12.4).

### Prerequisites
* NVIDIA GPU with CUDA 12.4 compatible drivers.
* [Conda](https://docs.anaconda.com/miniconda/) or Miniconda installed.

### Option A: Run via Docker
1. Build the image: `docker build -t anomaly-annotator .`
2. Run the container: `docker run --gpus all -p 8501:8501 anomaly-annotator`
3. Open your browser and navigate to `http://localhost:8501`

### Option B: Local Development via Conda
1. Clone repository: 
	'''bash
	git clone [https://github.com/anjgela/anomaly_annotation_system.git](https://github.com/anjgela/anomaly_annotation_system.git)
   cd anomaly_annotation_system'
2. Environment setup: 	
	'''bash
	conda env create -f environment.yml
	conda activate anomaly_env
	cp .env.example .env
3. System setup:
	'''bash
	chmod +x setup.sh
	./setup.sh

6. Application launch:
	'''bash
	streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false

## Models

* GroundingDINO
* SAM 2.1 (Hiera Tiny)
* YOLOE-S6L-Seg
* SAM 3