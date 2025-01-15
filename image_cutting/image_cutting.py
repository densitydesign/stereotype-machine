#THIS IS image_cutting.PY, currently its the state-of-the-art

#Add export PYTORCH_ENABLE_MPS_FALLBACK=1 if bugging w/ mps

import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import warnings
from PIL import UnidentifiedImageError
import sys
from typing import List, Tuple
import logging
from multiprocessing import Pool, Manager

USE_SAM2 = True

# Adding paths to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
emif_maskingdino_path = os.path.join(current_dir, '..', 'EMIF_MASKINGDINO')
efficientvit_path = os.path.join(current_dir, 'efficientvit')
sam_hq_path = os.path.join(current_dir, 'sam-hq')

sys.path.append(efficientvit_path)
sys.path.append(sam_hq_path)
sys.path.append(emif_maskingdino_path)

from groundingdino.util.inference import load_model, load_image # type: ignore
from groundingdino.util.inference_on_a_image import get_grounding_output # type: ignore
import groundingdino.datasets.transforms as T # type: ignore
from samHq.segment_anything import sam_model_registry, SamPredictor # type: ignore
from config.text_prompts import text_prompts

from torchvision.ops import box_convert

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*annotate is deprecated*")

################## DEVICE SELECTION ##################

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

TEXT_THRESHOLD = 0.35
global_folder = '/Users/tommasoprinetti/Desktop/ROOT_FOLDER'
model_folder = '/Users/tommasoprinetti/Documents/EMIF_REHARSAL/ROOT/EMIF_MASKINGDINO/model_folder'

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

################## CORE FUNCTIONS ##################

def worker_init():
    """Initializer for each worker process."""
    global global_model, global_device
    if torch.cuda.is_available():
        global_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        global_device = torch.device('mps')
    else:
        global_device = torch.device('cpu')

    global_model = load_model(
        "/Users/tommasoprinetti/Documents/EMIF_REHARSAL/ROOT/EMIF_MASKINGDINO/weights/GroundingDINO_SwinB_cfg.py",
        f"{model_folder}/groundingdino_swinb_cogcoor.pth",
        device=global_device
    )

def createBoxes(image_path: str, text_prompt: str, box_threshold: float):
    global global_model, global_device
    image_source, image = load_image(image_path)
    boxes_filt, pred_phrases = get_grounding_output(
        model=global_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=TEXT_THRESHOLD,
        cpu_only=global_device,
        token_spans=None
    )

    h, w, _ = image_source.shape
    boxes_filt = boxes_filt.cpu()
    boxes_xyxy = boxes_filt * torch.tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    return boxes_xyxy, image_source

def extractImages(boxes_xyxy, image_path: str, text_prompt: str, output_folder: str, bypass_filling=False):
    device = global_device
    model_type = "vit_h"

    sam_checkpoint = f"{model_folder}/sam_hq_vit_h.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    sam.to(device=device)
    predictor = SamPredictor(sam)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    if boxes_xyxy.size == 0:
        tqdm.write(f"No boxes found for image {image_path}. Printing null box.")
        os.makedirs(output_folder, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        image_output_folder = os.path.join(output_folder, base_filename)
        os.makedirs(image_output_folder, exist_ok=True)
        prompt_word = next((word for word in text_prompt.split() if len(word) > 3), "prompt")
        bw_mask = np.zeros((2048, 2048), dtype=np.uint8)
        bw_mask_output_path = os.path.join(image_output_folder, f"NULL_{base_filename}_{prompt_word}_mask.png")
        cv2.imwrite(bw_mask_output_path, bw_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        tqdm.write("NULL mask saved to:", bw_mask_output_path)
        return

    input_boxes = torch.tensor(boxes_xyxy, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

    with torch.no_grad():
        masks_refined, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            mask_input=None,
            multimask_output=False,
            return_logits=False,
            hq_token_only=False
        )

        masks_refined = masks_refined.cpu().numpy()
        masks_refined = masks_refined.squeeze(1)
        true_false_mask = np.any(masks_refined, axis=0)
        grayscale_mask = true_false_mask.astype(np.uint8) * 255

    if bypass_filling:
        bw_mask = grayscale_mask.astype(np.uint8)
        
    else:
        contours, _ = cv2.findContours(grayscale_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(grayscale_mask, [contour], 0, 255, -1)

        filled_mask_with_contours = grayscale_mask.copy()
        
        # Parameters
        kernel_size = 10
        blur_kernel_size = 5
        
        # Create structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        filled_mask_with_contours = cv2.morphologyEx(filled_mask_with_contours, cv2.MORPH_OPEN, kernel)
        filled_mask_with_contours = cv2.morphologyEx(filled_mask_with_contours, cv2.MORPH_CLOSE, kernel)
        filled_mask_with_contours = cv2.GaussianBlur(filled_mask_with_contours, (blur_kernel_size, blur_kernel_size), 0)

        filled_mask = cv2.bitwise_not(filled_mask_with_contours)
        final_mask = cv2.bitwise_not(filled_mask)
        bw_mask = final_mask.astype(np.uint8)

    #Measure mask
    height, width = bw_mask.shape[:2]
    print("Final Mask dimensions:", width, "x", height)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Create a dedicated folder for each image
    image_output_folder = os.path.join(output_folder, base_filename)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder, exist_ok=True)


    # Extract the first word with more than three characters from TEXT_PROMPT
    prompt_word = next((word for word in text_prompt.split() if len(word) > 3), "prompt")

    # Save the B/W mask image
    bw_mask_output_path = os.path.join(image_output_folder, f"{base_filename}_{prompt_word}_mask.png")
    cv2.imwrite(bw_mask_output_path, bw_mask, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    #print("B/W mask saved to:", bw_mask_output_path)

def worker_process_image(args):
    image_path, text_prompt, box_threshold, output_folder = args
    try:
        boxes_xyxy, annotated_frame = createBoxes(image_path, text_prompt, box_threshold)
        extractImages(boxes_xyxy, image_path, text_prompt, output_folder)
        logging.info(image_path)
        return True
    except UnidentifiedImageError:
        print(f"Cannot identify image file {image_path}. Skipping.")
        return False
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False
    
def get_last_processed_image(log_file):
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                return last_line
    except FileNotFoundError:
        return None

def process_images(root_folder, output_folder, start_from_zero=True):

    if not os.path.exists(global_folder):
        raise FileNotFoundError(f"Global folder not found: {global_folder}")

    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"Model folder not found: {model_folder}")

    log_file = f'{global_folder}/process_log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO)

    # If start_from_zero is True, erase the log file
    if start_from_zero:
        open(log_file, 'w').close()
        last_processed_image = None
    else:
        last_processed_image = get_last_processed_image(log_file)

    # Build the list of tasks
    tasks = []
    for text_prompt, box_threshold in text_prompts.items():
        for subdir, _, files in os.walk(root_folder):
            files.sort()
            # Count how many images total
            valid_images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]

            for file in valid_images:
                input_image_path = os.path.join(subdir, file)
                if last_processed_image and input_image_path <= last_processed_image:
                    continue
                relative_path = os.path.relpath(subdir, root_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                tasks.append((input_image_path, text_prompt, box_threshold, output_subfolder))

    # Process tasks with multiprocessing
    with Pool(processes=4, initializer=worker_init) as pool:
        results = pool.imap_unordered(worker_process_image, tasks)
        with tqdm(total=len(tasks), desc="Processing Images") as pbar:
            for _ in results:
                pbar.update(1)

    # Ensure the log file exists if it didn't before
    if last_processed_image is None:
        open(log_file, 'a').close()

if __name__ == "__main__":
    print("Your current device is:", device)
    root_folder = f'/Users/tommasoprinetti/Desktop/Upscaled_Images'
    output_folder = f'{global_folder}/OUTPUT_MASKS'
    process_images(root_folder, output_folder, start_from_zero=True)