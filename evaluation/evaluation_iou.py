import numpy as np
import cv2
import os
import argparse

def load_yolo_polygon_to_mask(txt_path, img_width, img_height):

    #generates empty black mask (0)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if not os.path.exists(txt_path):
        return mask #empty if file does not exists (no detection)
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        #part[0] class, then x,y coordinates
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        
        #denormalising into real pixels
        coords[:, 0] *= img_width
        coords[:, 1] *= img_height
        
        #converison (integers) for opencv
        poly_coords = np.int32(coords)
        
        #draw filled polygon on mask (1)
        cv2.fillPoly(mask, [poly_coords], 1)
        
    return mask

def calculate_iou(mask1, mask2):

    #1 pixels in both masks (and)
    intersection = np.logical_and(mask1, mask2).sum()
    
    #1 pixels in at least one mask (or)
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0 #avoid division by zero if both are empty
        
    return intersection / union

if __name__ == "__main__":
    
    #argument configuration (terminal)
    parser = argparse.ArgumentParser(description="Calculate IoU between Gorund Truth and model prediction.")
    parser.add_argument("--video", type=str, required=True, help="Video file path")
    parser.add_argument("--gt", type=str, required=True, help="Groung Turth text file path")
    parser.add_argument("--model", type=str, required=True, help="Model text file path")
    args = parser.parse_args()

    #extracting video dimensions
    cap = cv2.VideoCapture(args.video) #open video
    if cap.isOpened(): #check if video has been opened
        WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #converting to int
        HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        print(f"Error: video '{args.video}' not opened correctly.")
        exit(1)

    cap.release() #release
    
    #create masks
    mask_gt = load_yolo_polygon_to_mask(args.gt, WIDTH, HEIGHT)
    mask_model = load_yolo_polygon_to_mask(args.model, WIDTH, HEIGHT)
    
    #calculate iou
    iou = calculate_iou(mask_gt, mask_model)

    print(f"Analysed video: {args.video}")
    print(f"Ground Truth file: {args.gt}")
    print(f"Model file: {args.model}")
    print(f"IoU score: {iou:.4f} ({iou*100:.2f}%)")