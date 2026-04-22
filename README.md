# SEMI-AUTOMATIC ANOMALY ANNOTATION SYSTEM IN THERMAL VIDEOS
This is an interactive application provided with a web UI via the Streamlit framework, developed as a Bachelor's degree reaserch thesis.

## Installation and Execution

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
3. Environment setup:

   '''bash
	
	conda env create -f environment.yml
	
	conda activate anomaly_env
	
	cp config.txt .env

4. System setup:
	
	'''bash
	
	chmod +x setup.sh
	
	./setup.sh

6. Application launch:
	
	'''bash
	
	streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false

## Models

* RFI Anomaly Detection [https://github.com/Aleinno001/RFI_anomaly_detection](https://github.com/Aleinno001/RFI_anomaly_detection)
	* GroundingDINO
	* SAM 2.1 (Hiera Tiny)
* YOLOE-S6L-Seg
* SAM 3

## Models Evaluation (IoU)
The evaluation_iou.py script has been provided to evaluate the models' performance using the Intersection over Union metrics.

The "evaluation" folder provides the script and a few example files to run the test.

### Execution
To execute the script using the demo files provided (frame 300):
* SAM 3 using Promptable Concept Segmentation:

  '''bash

  python evaluation_iou.py --video test_video_san_donato_Trim.mp4 --gt frame_300_gt.txt --model frame_300_sam_pcs.txt

* SAM 3 using Promptable Visual Segmentation:

  '''bash
  python evaluation_iou.py --video test_video_san_donato_Trim.mp4 --gt frame_300_gt.txt --model frame_300_sam_pvs.txt

* YOLOE-26L-Seg:

  '''bash

  python evaluation_iou.py --video test_video_san_donato_Trim.mp4 --gt frame_300_gt.txt --model frame_300_yolo.txt

In order to execute the script using different files: execute the commands listed above changing the --video, --gt and --model arguments.
