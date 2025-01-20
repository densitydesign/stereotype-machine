#image_cutting_sam2.py

import torch
from ultralytics import SAM
from ultralytics.models.sam import Predictor as SAMPredictor

import os
import cv2
import time
import numpy as np
from tqdm import tqdm
import warnings
from PIL import UnidentifiedImageError
import sys
from typing import List
import logging
from PIL import Image
from multiprocessing import Pool
from PIL import ImageDraw
from torchvision.ops import box_convert

from scipy.ndimage import label

# Adding paths to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.append(project_root)

from GroundingDINO.groundingdino.util.inference import load_model, load_image
from GroundingDINO.demo.inference_on_a_image import get_grounding_output

TEXT_THRESHOLD = 0.35
model_folder = f"{project_root}/model_folder"

#sam_model = SAM(f"{model_folder}/sam_l.pt")

mode = "predict"

################## DEVICE SELECTION ##################

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

################## CORE FUNCTIONS ##################

def initialize_sam_model():
    """
    Initialize the SAM model, downloading it if necessary.
    """     
    sam_model_filename = "sam_l.pt"
    sam_model = f"{project_root}/model_folder/{sam_model_filename}"
    sam_download_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_l.pt"

    # Check if the SAM model exists, and download if missing
    if not os.path.exists(sam_model):
        print(f"🚨 SAM model not found at {sam_model}. Downloading...")
        download_file(sam_model, sam_download_url)
        print(f"✅ SAM model downloaded to {sam_model}.")
        
    else:
        print(f"✅ SAM model found at {sam_model}.")

    # Initialize and return the SAM model
    return sam_model

def download_file(file_path: str, download_url: str) -> None:
    """
    Download a file from a URL and save it to the specified file_path.
    If the download fails, the function will print an error and exit.
    """
    try:
        import requests  # Ensure requests is available
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Raise an exception for any errors
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {file_path}")
    except Exception as e:
        print(f"🚨 Error downloading {file_path} from {download_url}: {e}")
        sys.exit(1)

def worker_init():
    """Initializer for each worker process."""
    global global_model, global_device
    if torch.cuda.is_available():
        global_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        global_device = torch.device('mps')
    else:
        global_device = torch.device('cpu')

    model_filename = "groundingdino_swinb_cogcoor.pth"
    cfg_filename   = "GroundingDINO_SwinB_cfg.py"
    model_path = f"{model_folder}/{model_filename}"
    cfg_path   = f"{model_folder}/{cfg_filename}"

    DOWNLOAD_PTH = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    DOWNLOAD_CFG = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py"

    # Download the model file if not present
    if not os.path.exists(model_path):
        print(f"🚨 Model file not found at {model_path}. Downloading from {DOWNLOAD_PTH} ...")
        download_file(model_path, DOWNLOAD_PTH)

    else:
        print("🟢 Model file found, going on:")

    # Download the configuration file if not present
    if not os.path.exists(cfg_path):
        print(f"🚨 Config file not found at {cfg_path}. Downloading from {DOWNLOAD_CFG} ...")
        download_file(cfg_path, DOWNLOAD_CFG)

    else:
        print("🟢 Model file found, going on:")


    global_model = load_model(cfg_path, model_path, device=global_device)

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

def extractImages(boxes_xyxy, sam_model, image_path: str, text_prompt: str, output_folder: str, bypass_filling=False):
    # Debug: Print inputs at the beginning
    print("=== Debug: Entering extractImages ===")
    print(f"Image path: {image_path}")
    print(f"Text prompt: {text_prompt}")
    print(f"Output folder: {output_folder}")
    print(f"Bypass filling: {bypass_filling}")
    print(f"Received boxes shape/type: {boxes_xyxy.shape if hasattr(boxes_xyxy, 'shape') else type(boxes_xyxy)}")
    
    if len(boxes_xyxy) == 0:
        print(f"[DEBUG] No bounding boxes provided for {image_path}. Skipping.")
        return

    # 2. Ensure the output folder exists
    if not os.path.exists(output_folder):
        print(f"[DEBUG] Output folder '{output_folder}' does not exist. Creating folder.")
        os.makedirs(output_folder)
    else:
        print(f"[DEBUG] Output folder '{output_folder}' already exists.")

    # 3. Create a SAMPredictor with some global overrides
    overrides = dict(
        conf=0.2,               # Confidence threshold
        mode=mode,              # Ensures we do inference 
        imgsz=1024,             # Higher resolution may yield finer masks
        model=f"{sam_model}",   # If needed, or use a different checkpoint
        save_dir=f"{project_root}/Raw_predicts"
    )
    print("[DEBUG] SAMPredictor overrides:", overrides)
    
    predictor = SAMPredictor(overrides=overrides)
    print("[DEBUG] Created SAMPredictor.")

    try:
        # 4. Load the image into the predictor
        print(f"[DEBUG] Loading image into predictor: {image_path}")
        predictor.set_image(image_path)

        # 5. Call predictor based on mode
        if mode == "predict":
            print(f"[DEBUG] Running SAM in 'predict' mode on {image_path} with boxes:")
            print(boxes_xyxy)
            results = predictor(
                bboxes=boxes_xyxy,
                points_stride=64, 
                crop_n_layers=1, 
            )
        elif mode == "segment":
            print(f"[DEBUG] Running SAM in 'segment' mode on {image_path}...")
            results = predictor()

        print("[DEBUG] Results type:", type(results))
        if isinstance(results, list):
            print(f"[DEBUG] Number of result objects: {len(results)}")
            for idx, r in enumerate(results):
                has_masks = hasattr(r, 'masks') and (r.masks is not None)
                print(f"[DEBUG] Result {idx + 1}: Masks available: {has_masks}")
        else:
            print("[DEBUG] Results is not a list. Type:", type(results))

    except Exception as e:
        print(f"🚨 Error running SAM on {image_path}: {e}")
        sys.exit(1)

    # Helper function: merges masks from a single Results object into combined_mask
    def combine_masks_from_results(results_obj, combined_mask):
        print("[DEBUG] Combining masks from one result object...")
        if results_obj.masks is not None and hasattr(results_obj.masks, "data"):
            for idx, mask_tensor in enumerate(results_obj.masks.data):
                try:
                    mask_numpy = mask_tensor.cpu().numpy()
                    print(f"[DEBUG] Mask {idx+1} shape: {mask_numpy.shape}")
                except Exception as conv_ex:
                    print(f"🚨 Error converting mask to numpy: {conv_ex}")
                if combined_mask is None:
                    combined_mask = mask_numpy
                else:
                    combined_mask = np.logical_or(combined_mask, mask_numpy)
        else:
            print(f"[DEBUG] No masks detected for {image_path}.")
        return combined_mask

    # 6. Merge all masks into a single combined_mask
    combined_mask = None
    if isinstance(results, list):
        print("[DEBUG] Merging masks from multiple results objects...")
        for r in results:
            combined_mask = combine_masks_from_results(r, combined_mask)
    else:
        print("[DEBUG] Merging masks from a single result object...")
        combined_mask = combine_masks_from_results(results, combined_mask)

    # 7. Optional: fill holes
    def fill_holes(binary_mask):
        """
        A more robust hole-filling method using connected component analysis.
        """
        try:
            binary_mask = binary_mask.astype(bool)
            inverted_mask = ~binary_mask
            labeled_holes, num_features = label(inverted_mask)
            print(f"[DEBUG] fill_holes: {num_features} features found.")
            filled_mask = binary_mask.copy()
            for i in range(1, num_features + 1):
                hole = (labeled_holes == i)
                # If the hole doesn't touch the boundary, fill it
                if not (hole[0, :].any() or hole[-1, :].any() or hole[:, 0].any() or hole[:, -1].any()):
                    filled_mask = np.logical_or(filled_mask, hole)
            return filled_mask.astype(np.uint8)
        except Exception as e:
            print(f"🚨 Error in fill_holes: {str(e)}")
            return binary_mask.astype(np.uint8)

    # 8. Save the combined mask
    if combined_mask is not None:
        print("[DEBUG] Combined mask computed. Proceeding to fill holes (if not bypassed).")
        combined_mask_filled = combined_mask if bypass_filling else fill_holes(combined_mask)
        print("[DEBUG] Combined mask filled. Converting to grayscale.")
        combined_mask_grayscale = (combined_mask_filled.astype(np.uint8) * 255)
        combined_mask_image = Image.fromarray(combined_mask_grayscale)
        combined_output_path = os.path.join(
            output_folder,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_combined_mask_{text_prompt}.png"
        )
        combined_mask_image.save(combined_output_path)
        print(f"[DEBUG] Saved combined mask to {combined_output_path}")
    else:
        print(f"[DEBUG] No masks to combine for {image_path}.")

    print("=== Debug: Exiting extractImages ===\n")

def worker_process_image(args):
    image_path, text_prompt, box_threshold, output_folder, sam_model = args
    try:
        log_msg = f"Processing {image_path} with tag '{text_prompt}' and threshold {box_threshold}"
        if log_callback := globals().get("log_callback"):
            log_callback(log_msg)
        else:
            print(log_msg)

        boxes_xyxy, _ = createBoxes(image_path, text_prompt, box_threshold)
        print("These are the boxes:", boxes_xyxy)
        print("🟢 Forwarding to extractImages:")

        extractImages(boxes_xyxy, sam_model, image_path, text_prompt, output_folder)

        logging.info(f"Task for {image_path} completed successfully.")
        return True
    except Exception as e:
        error_msg = f"🚨 Task failed for {image_path}: {e}"
        logging.error(error_msg, exc_info=True)
        if log_callback := globals().get("log_callback"):
            log_callback(error_msg)
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

def process_images(input_folder, output_folder, sam_model, start_from_zero=True, selected_tags=None, log_callback=None, progress_callback=None):

    if log_callback is None:
        log_callback = print

    if selected_tags is None:
        selected_tags = {}

    log_file = os.path.join(project_root, 'process_log.txt')
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # If start_from_zero is True, erase the log file
    if start_from_zero:
        open(log_file, 'w').close()
        last_processed_image = None
    else:
        last_processed_image = get_last_processed_image(log_file)

    tasks = []

    for text_prompt, box_threshold in selected_tags.items():
        for subdir, _, files in os.walk(input_folder):
            files.sort()
            valid_images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]

            for file in valid_images:
                input_image_path = os.path.join(subdir, file)
                if last_processed_image and input_image_path <= last_processed_image:
                    continue
                relative_path = os.path.relpath(subdir, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
              
                tasks.append((input_image_path, text_prompt, box_threshold, output_subfolder, sam_model))

    total_tasks = len(tasks)
    log_callback(f"Total images to process: {total_tasks}")

    # Process tasks with multiprocessing.
    with Pool(processes=1, initializer=worker_init) as pool:
        results = pool.imap_unordered(worker_process_image, tasks)
        with tqdm(total=total_tasks, desc="Processing Images") as pbar:
            for i, result in enumerate(results, start=1):
                if result:
                    log_callback("✅ Task completed successfully.")
                else:
                    log_callback("❌ Task failed. See log for details.")
                pbar.update(1)
                if progress_callback is not None:
                    progress_callback(i, total_tasks)

    if last_processed_image is None:
        open(log_file, 'a').close()