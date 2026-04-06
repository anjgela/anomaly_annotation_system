import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" #fixing pytorch fragmentation

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
from ultralytics import YOLO, SAM
import glob
import shutil
import subprocess

def extract_anomalous_frames(file_path):
    full_path = os.path.join("RFI_anomaly_detection", file_path)
    if not os.path.exists(full_path):
      st.warning("No anomaly file produced.")
      return []
    with open(full_path, 'r') as file:
        rows = file.readlines()
        frames_list = [int(row.strip()) for row in rows if row.strip().isdigit()]
        return frames_list

st.set_page_config(layout="wide", page_title="Thermal Anomalies Annotation")
st.title("Anomaly Annotation System for Thermal Videos")

#session_state inisialisation
#creating two states (og, ann)
def init_state(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value

#save clicks on original uploaded video
init_state('positive_seeds_og', {})
init_state('negative_seeds_og', {})
init_state('last_click_og', None)

#save clicks on annotated video
init_state('positive_seeds_ann', {})
init_state('negative_seeds_ann', {})
init_state('last_click_ann', None)

#differentiating selected video
init_state('og_video_path', None)
init_state('ann_video_path', None)
init_state('selected_video', "original")

#counting frames for slider
init_state('total_frames', 0)

#save pre-processed frames
init_state('suspect_frames', [])
init_state('current_frame', 0)

#outputs
init_state('output_video', None)
init_state('output_zip', None)

#callback for changes on uploaded file
def reset_state():
    #original video 
    st.session_state.positive_seeds_og = {}
    st.session_state.negative_seeds_og = {}
    st.session_state.last_click_og = None
    #annotated video
    st.session_state.positive_seeds_ann = {}
    st.session_state.negative_seeds_ann = {}
    st.session_state.last_click_ann = None
    #outputs
    st.session_state.output_video = None
    st.session_state.output_zip = None
    #frames
    st.session_state.total_frames = 0
    st.session_state.suspect_frames = []
    st.session_state.current_frame = 0
    #video selection
    st.session_state.og_video_path = None
    st.session_state.ann_video_path = None
    st.session_state.selected_video = "original"

#sidebar menu
with st.sidebar:
    st.header("MENU")
    uploaded_video = st.file_uploader("Upload thermal video (.mp4)",
        type=['mp4'],
        on_change = reset_state)

    #reset annotations
    if st.button("Reset Annotations"):
        st.session_state.positive_seeds_og = {}
        st.session_state.negative_seeds_og = {}
        st.session_state.positive_seeds_ann = {}
        st.session_state.negative_seeds_ann = {}
        st.session_state.last_click_og = None
        st.session_state.last_click_ann = None
        st.rerun()

    #download video
    if st.session_state.output_video is not None:
      clean_name = os.path.splitext(uploaded_video.name)[0]
      st.download_button(
          label="Download annotated video (.mp4)",
          data=st.session_state.output_video,
          file_name=clean_name+"_annotated.mp4",
          mime="video/mp4",
          on_click="ignore",
          icon=":material/download:",
          icon_position="right",
          width="stretch"
      )
    else:
      st.download_button(
              label="Download annotated video (.mp4)",
              data="dummy",
              icon=":material/download:",
              icon_position="right",
              disabled=True,
              width="stretch"
          )

    #donwload zip
    if st.session_state.output_zip is not None:
      clean_name = os.path.splitext(uploaded_video.name)[0]
      st.download_button(
            label="Download annotations (.zip)",
            data=st.session_state.output_zip,
            file_name=clean_name+"_labels.zip",
            mime="application/zip",
            on_click="ignore",
            icon=":material/download:",
            icon_position="right",
            width="stretch"
        )
    else:
        st.download_button(
                label="Download annotations (.zip)",
                data="dummy",
                icon=":material/download:",
                icon_position="right",
                disabled=True,
                width="stretch"
        )

#extracting frame when video is uploaded (original)
frame_rgb_og  = None 
frame_rgb_ann = None 
if uploaded_video is not None and st.session_state.og_video_path is None:
        uploaded_video.seek(0) #rewind to initial frame

        #creating temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        st.session_state.og_video_path = tfile.name

        cap = cv2.VideoCapture(st.session_state.og_video_path) #opening the video file
        st.session_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #save total frames
        cap.release()


col_img, col_ctrl = st.columns([7, 3]) #70% image, 30% controls

#controls: right column
with col_ctrl:
    st.header("Settings")

    #anomaly detection (og)
    st.subheader("Pre-processing (AD)")
    if st.button(label="Start RFI Anomaly Detection", width="stretch"):
        if st.session_state.og_video_path is None:
            st.error("Upload video (MENU) o start detection.")
        else:
            with st.spinner(text="RFI anomaly detection in progress..."):
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONPATH"] = "/workspace"
            
                #dynamically finds PyTorch lybraries and system lybraries (CUDA 13.0) 
                import site
                nvrtc_paths = [os.path.join(p, "nvidia/cuda_nvrtc/lib") for p in site.getsitepackages()]
                cuda_paths = ["/usr/local/cuda/lib64", "/usr/local/cuda-13.0/lib64"]
                env["LD_LIBRARY_PATH"] = ":".join(nvrtc_paths + cuda_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

                ad_results = subprocess.Popen(
                    ["python", "-u", "anomaly_detection.py", "--input_video", st.session_state.og_video_path], #u: unbuffered output
                    cwd="/workspace/RFI_anomaly_detection",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env
                )
                progress_bar = st.progress(0)
                progress_text = st.empty()
                total_frames = None

                while True:
                    line = ad_results.stdout.readline()
                    if not line:
                        if ad_results.poll() is not None:
                            break
                        time.sleep(0.1)
                        continue
                    line = line.strip()
                    st.write(line) #debug
        
                    if line.startswith("TOTAL_FRAMES:"):
                        total_frames = int(line.split(":")[1])
        
                    elif line.startswith("PROGRESS:") and total_frames:
                        current= int(line.split(":")[1])
                        progress_bar.progress(min(current/total_frames, 1.0))
                        progress_text.write(f"Processing frame {current}/{total_frames}")

                ad_results.wait()

                #error handling
                stderr = ad_results.stderr.read()
                if stderr.strip():
                    st.error("RFI Anomaly Detection has generated an error:")
                    st.code(stderr)
                    st.stop() #to render message visible and pause site until new user's input

                st.session_state.suspect_frames = extract_anomalous_frames("found_anomalies.txt")
                if len(st.session_state.suspect_frames) > 0:
                    st.session_state.current_frame = st.session_state.suspect_frames[0]
                    st.success("Anomalies found!")
                    time.sleep(3) #to render message visible
                else:
                    st.info("No anomalies detected.")
                    time.sleep(3) #to render message visible
                st.rerun()

    st.divider()

    #choosing the model
    model_choice = st.selectbox(
        "Select the model:",
        ["YOLOE-26L-Seg", "SAM 3"]
    )

    yolo_prompt = ""
    if model_choice == "YOLOE-26L-Seg":
        yolo_prompt = st.text_input("Enter objects to detect (comma separated):", value="person, animal")
        #st.caption("Example: person, dog, bright object, hot silhouette")

    st.divider()

    #HITL
    st.subheader("Navigation and Cursor")

    #frame navigator among suspect frames (from AD)
    if len(st.session_state.suspect_frames) > 0:
        selector = st.selectbox(
            label="Select suspect frame to view:",
            options=st.session_state.suspect_frames,
            index=st.session_state.suspect_frames.index(st.session_state.current_frame) if st.session_state.current_frame in st.session_state.suspect_frames else 0
        )
        if selector != st.session_state.current_frame:
            st.session_state.current_frame = selector
            st.rerun()
    #manual frame navigator
    if st.session_state.total_frames > 0:
      manual_frame = st.slider(
          label="Scroll video manually",
          min_value=0,
          max_value=st.session_state.total_frames-1,
          value=st.session_state.current_frame,
      )
      if manual_frame != st.session_state.current_frame:
            st.session_state.current_frame = manual_frame
            st.rerun()
    else:
        st.write("Showing frame 0.")

    #choosing seed (+ive or -ive)
    seed_type = st.radio(
        label="Seed:",
        options=["Positive", "Negative"]
    )

    st.divider()

    #select video (original or annotated)
    st.subheader("Manual Annotation")
    st.info(f"Selected video: **{st.session_state.selected_video}**\n(Click on other image to switch)")
    
    #visualising clicked coordinates
    st.write(f"**Coordinates on {st.session_state.selected_video} video:**")
    if st.session_state.selected_video == "original":
        st.write(f"Positives: {st.session_state.positive_seeds_og}")
        st.write(f"Negatives: {st.session_state.negative_seeds_og}")
    if st.session_state.selected_video == "processed":
        st.write(f"Positives: {st.session_state.positive_seeds_ann}")
        st.write(f"Negatives: {st.session_state.negative_seeds_ann}")

    st.divider()

    #execution
    st.subheader("Processing")
    if st.button("Process video", width="stretch", type="primary"):
        if st.session_state.og_video_path is None:
            st.error("Upload a video from the MENU first.")
        else:
            target_video = st.session_state.og_video_path if st.session_state.selected_video == "original" else st.session_state.ann_video_path
            target_seeds = st.session_state.positive_seeds_og if st.session_state.selected_video == "original" else st.session_state.positive_seeds_ann
        
            if model_choice == "SAM 3" and len(target_seeds) == 0:
                st.error("SAM 3 requires at least one Positive Seed to start.")
            else:
                with st.spinner(text=f"Processing '{st.session_state.selected_video}' video with {model_choice}..."):
                    #setup for progress bar and video extraction
                    cap = cv2.VideoCapture(target_video)
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
    
                    if model_choice == "YOLOE-26L-Seg":
                        cap.release()
                        path_yolo = "/workspace/realtime-detection-yolo26/yoloe-26l-seg.pt"
                        model = YOLO(path_yolo)
    
                        #open vocabulary: defining classes to find
                        custom_classes = [c.strip() for c in yolo_prompt.split(",") if c.strip()]
                        if custom_classes:
                            model.set_classes(custom_classes)
    
                        results = model.predict(
                            source=target_video,
                            conf=0.15, #lowered confidence to allow it to show masks
                            save=True, #save video
                            save_txt=True, #save annotations in standard YOLO format (.txt)
                            project="/workspace/results",
                            name="yolo_out",
                            exist_ok=True,
                            stream=True #for progress bar
                          )
    
                        #updating progress bar
                        processed_frames = 0
                        for frame_result in results:
                            processed_frames += 1
                            if processed_frames % 10 == 0 or processed_frames == st.session_state.total_frames:
                                progress_bar.progress(min(processed_frames / st.session_state.total_frames, 1.0))
                                progress_text.write(f"YOLO is scanning frame {processed_frames} of {st.session_state.total_frames}...")
    
                    elif model_choice == "SAM 3":
                        import torch
                        from sam2.build_sam import build_sam2_video_predictor
        
                        #GPU (RTX 4XXX) optimisation and memory clean up
                        torch.cuda.empty_cache() #freeing VRAM of old precesses
                        torch.backends.cuda.matmul.allow_tf32 = True #speeding up calculations on RTX 4000
                        torch.backends.cudnn.allow_tf32 = True
    
                        #delete old SAM masks
                        sam_out_path = "/workspace/results/sam_out"
                        if os.path.exists(sam_out_path):
                            shutil.rmtree(sam_out_path)
                        os.makedirs(sam_out_path, exist_ok=True)
    
                        clicked_frames = [f for f, seeds in target_seeds.items() if len(seeds) > 0]
    
                        if len(clicked_frames) > 0:
    
                            #extracting frames
                            temp_frames_dir = "/workspace/temp_stateful_frames"
                            if os.path.exists(temp_frames_dir):
                                shutil.rmtree(temp_frames_dir)
                            os.makedirs(temp_frames_dir, exist_ok=True)
                          
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_count = 0
                            while True:
                                ret, frame = cap.read()
                                if not ret: 
                                    break
                                cv2.imwrite(os.path.join(temp_frames_dir, f"{frame_count:05d}.jpg"), frame)
                                frame_count += 1
        
                            #memory initialisation (inference state)
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                      
                            #native motor and wwights of sam 2 (also used by sam3)
                            sam2_checkpoint = "/workspace/RFI_anomaly_detection/models/sam2.1/sam2.1_hiera_tiny.pt"
                            sam2_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
                            predictor = build_sam2_video_predictor(sam2_cfg, sam2_checkpoint, device=device)
    
                            #abilitating autocast not to let CUDNN crash
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                #inference state
                                inference_state = predictor.init_state(video_path=temp_frames_dir)
            
                                #uploading the seeds
                                obj_id_counter = 1
        
                                for f_idx in clicked_frames:
                                    clicks = [[p[0], p[1]] for p in target_seeds[f_idx]]
                                    seed_labels = [1] * len(clicks) #1 = positive seed
        
                                    #adding each annotated frame to the model's state 
                                    predictor.add_new_points_or_box(
                                        inference_state=inference_state, 
                                        frame_idx=f_idx, 
                                        obj_id=obj_id_counter,
                                        points=np.array(clicks, dtype=np.float32), 
                                        labels=np.array(seed_labels, dtype=np.int32),
                                    )
                                    #assigning progressive ID to each anomaly
                                    obj_id_counter += 1
        
                                #propagation
                                progress_text.write("Tracking anomalies...")
                        
                                final_video_path = os.path.join(sam_out_path, "stateful_tracked_anomalies.mp4")
                                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                out_final = cv2.VideoWriter(final_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                              
                                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                                    progress_bar.progress((out_frame_idx + 1) / frame_count)
        
                                    #taking original frame from the disk
                                    img_path = os.path.join(temp_frames_dir, f"{out_frame_idx:05d}.jpg")
                                    img = cv2.imread(img_path)
        
                                    #dynamic overlapping of the masks
                                    if len(out_mask_logits) > 0:
                                        color_mask = np.zeros_like(img)
                                        has_mask = False
        
                                        #iterating on all anomalies in current frame
                                        for i, obj_id in enumerate(out_obj_ids):
                                            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                                            if mask.sum() > 0:
                                                color_mask[mask] = [0, 0, 255] #red BGR
                                                has_mask = True
                                        
                                        if has_mask:
                                            img = cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)
                                            
                                    out_final.write(img)
          
                                out_final.release()
                                shutil.rmtree(temp_frames_dir) #deleting temporary frames
                            cap.release()
    
                    progress_text.success("Processing complete.")
    
                    #downloadble results at each iteration of the loop
                    output_folder = "/workspace/results/yolo_out" if model_choice == "YOLOE-26L-Seg" else "/workspace/results/sam_out"
                    generated_video_files = [f for f in glob.glob(f"{output_folder}/*") if f.endswith('.mp4') or f.endswith('.avi')] #video file output
                    
                    if generated_video_files:
                        recent = max(generated_video_files, key=os.path.getctime)
                        latest_ann = "/workspace/results/latest_annotated_loop.mp4"
                        os.makedirs("/workspace/results", exist_ok=True)
                        shutil.copy(recent, latest_ann)
                      
                        st.session_state.ann_video_path = latest_ann
                        with open(latest_ann, "rb") as video_file:
                            st.session_state.output_video = video_file.read()
                          
                        #set source of latest annotated video
                        st.session_state.selected_video = "processed"

                        #eliminate old seeds
                        st.session_state.positive_seeds_ann = {}
                        st.session_state.negative_seeds_ann = {}
                    else:
                        st.session_state.output_video = None
    
                    #zip dataset containing images and labels
                    if os.path.exists(output_folder):
                        path_zip = "/workspace/results/annotations_dataset"
                        shutil.make_archive(path_zip, 'zip', output_folder)
                        with open(path_zip + ".zip", "rb") as zip_file:
                            st.session_state.output_zip = zip_file.read()
                
                st.rerun()

#image: left column
with col_img:
    cf = st.session_state.current_frame
    st.header(f"Frame {cf}")

    if st.session_state.og_video_path is not None:
        #original video
        cap_og = cv2.VideoCapture(st.session_state.og_video_path)
        cap_og.set(cv2.CAP_PROP_POS_FRAMES, cf)
        ret_og, frame_og = cap_og.read()
        cap_og.release()

        if ret_og:
            st.subheader("Original Video")
            #selecting original video
            if st.session_state.selected_video == "original":
                st.success("Video selected for annotation")
            else:
                if st.button("Select original video", width="content", key="btn_sel_og"):
                    st.session_state.selected_video = "original"
                    st.rerun()
            img_to_draw_og = cv2.cvtColor(frame_og, cv2.COLOR_BGR2RGB) #image with planted seed saved

            #create seeds list for current frame
            if cf not in st.session_state.positive_seeds_og: 
                st.session_state.positive_seeds_og[cf] = []
            if cf not in st.session_state.negative_seeds_og: 
                st.session_state.negative_seeds_og[cf] = []

            for p in st.session_state.positive_seeds_og[cf]: 
                cv2.circle(img_to_draw_og, (p[0], p[1]), 5, (0, 255, 0), -1) #green
            for n in st.session_state.negative_seeds_og[cf]: 
                cv2.circle(img_to_draw_og, (n[0], n[1]), 5, (255, 0, 0), -1) #red

            value_og = streamlit_image_coordinates(Image.fromarray(img_to_draw_og), key=f"og_{cf}")

            #saving the coordinates in current frame
            if value_og is not None and value_og != st.session_state.last_click_og: 
                st.session_state.last_click_og = value_og
                st.session_state.selected_video = "original"
                point = (value_og["x"], value_og["y"])

                if "Positive" in seed_type:
                    if point not in st.session_state.positive_seeds_og[cf]: 
                        st.session_state.positive_seeds_og[cf].append(point)
                else:
                    if point not in st.session_state.negative_seeds_og[cf]: 
                        st.session_state.negative_seeds_og[cf].append(point)
                st.rerun()

        #annotated video
        if st.session_state.ann_video_path is not None:
            cap_ann = cv2.VideoCapture(st.session_state.ann_video_path)
            cap_ann.set(cv2.CAP_PROP_POS_FRAMES, cf)
            ret_ann, frame_ann = cap_ann.read()
            cap_ann.release()

            if ret_ann:
                st.divider()
                st.subheader("Processed Video")
                #selecting processed video
                if st.session_state.selected_video == "processed":
                    st.success("Video selected for annotation")
                else:
                    if st.button("Select processed video", width="content", key="btn_sel_ann"):
                        st.session_state.selected_video = "processed"
                        st.rerun()
                img_to_draw_ann = cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB)

                if cf not in st.session_state.positive_seeds_ann: 
                    st.session_state.positive_seeds_ann[cf] = []
                if cf not in st.session_state.negative_seeds_ann: 
                    st.session_state.negative_seeds_ann[cf] = []

                for p in st.session_state.positive_seeds_ann[cf]: 
                    cv2.circle(img_to_draw_ann, (p[0], p[1]), 5, (0, 255, 0), -1) 
                for n in st.session_state.negative_seeds_ann[cf]: 
                    cv2.circle(img_to_draw_ann, (n[0], n[1]), 5, (255, 0, 0), -1) 

                value_ann = streamlit_image_coordinates(Image.fromarray(img_to_draw_ann), key=f"ann_{cf}")

                if value_ann is not None and value_ann != st.session_state.last_click_ann: 
                    st.session_state.last_click_ann = value_ann
                    st.session_state.selected_video = "processed" # Sposta il Focus!
                    point = (value_ann["x"], value_ann["y"])

                    if "Positive" in seed_type:
                        if point not in st.session_state.positive_seeds_ann[cf]: 
                            st.session_state.positive_seeds_ann[cf].append(point)
                    else:
                        if point not in st.session_state.negative_seeds_ann[cf]: 
                            st.session_state.negative_seeds_ann[cf].append(point)
                    st.rerun()
                
    else:
        st.info("Upload a video (MENU) to start annotating.")
        
