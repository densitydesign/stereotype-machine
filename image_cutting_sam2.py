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

from scipy.ndimage import binary_fill_holes, generate_binary_structure, label


# Adding paths to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
emif_maskingdino_path = os.path.join(current_dir, '..', 'EMIF-MaskingDino')
efficientvit_path = os.path.join(current_dir, 'efficientvit')

sys.path.append(efficientvit_path)
sys.path.append(emif_maskingdino_path)

from groundingdino.util.inference import load_model, load_image # type: ignore
from groundingdino.util.inference_on_a_image import get_grounding_output # type: ignore
#from image_cutting.config.text_prompts import text_prompts

from torchvision.ops import box_convert

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TEXT_THRESHOLD = 0.35
global_folder = '/Users/tommasoprinetti/Desktop/ROOT_FOLDER'
model_folder = '/Users/tommasoprinetti/Documents/EMIF_REHARSAL/ROOT/EMIF-MaskingDino/model_folder'
sam_model = SAM(f"{model_folder}/sam_l.pt")

mode = "predict"

################## DEVICE SELECTION ##################

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

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
        "/Users/tommasoprinetti/Documents/EMIF_REHARSAL/ROOT/EMIF-MaskingDino/weights/GroundingDINO_SwinB_cfg.py",
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

def extractImages_old(boxes_xyxy, image_path: str, text_prompt: str, output_folder: str, bypass_filling=False):
    global sam_model

    if len(boxes_xyxy) == 0:
        print(f"No bounding boxes provided for {image_path}. Skipping.")
        return

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        #print(f"Created output folder: {output_folder}")
    
    # Run SAM2 inference with bounding boxes as prompts
    results_list = sam_model(image_path, bboxes=boxes_xyxy)

    # Check if the result is a list and handle it accordingly
    #print("\n=== Debug: Results from SAM2 inference ===")
    #print(f"Results list: {results_list}")

    combined_mask = None 

    if isinstance(results_list, list):
        for result_idx, result in enumerate(results_list):
            if result.masks is not None:
                #print(f"Found {len(result.masks.data)} masks for result {result_idx + 1} of {image_path}")
                for mask in result.masks.data:  # Use `data` attribute for tensor access
                    #print(f"Processing mask for result {result_idx + 1}")
                    #print(f"Mask type: {type(mask)}, shape: {mask.shape}")

                    # Convert the mask tensor to a NumPy array
                    mask_numpy = mask.cpu().numpy()

                    # Combine masks using logical OR
                    if combined_mask is None:
                        combined_mask = mask_numpy
                    else:
                        combined_mask = np.logical_or(combined_mask, mask_numpy)
            else:
                print(f"No masks detected for result {result_idx + 1} of {image_path}.")
    else:
        print("Unexpected result type. Expected a list of results.")

    def fill_holes(binary_mask):
        """
        A more robust hole-filling method using connected component analysis.

        Args:
            binary_mask (numpy.ndarray): Input binary mask.

        Returns:
            numpy.ndarray: Binary mask with holes filled.
        """
        try:
            # Ensure binary format
            binary_mask = binary_mask.astype(bool)

            # Invert the mask to identify holes
            inverted_mask = ~binary_mask

            # Label connected components in the inverted mask
            labeled_holes, num_features = label(inverted_mask)

            # Create a filled mask
            filled_mask = binary_mask.copy()

            # Iterate over connected components
            for i in range(1, num_features + 1):
                hole = (labeled_holes == i)

                # Check if the hole touches the boundary
                if not (hole[0, :].any() or hole[-1, :].any() or hole[:, 0].any() or hole[:, -1].any()):
                    # If the hole does not touch the boundary, fill it
                    filled_mask = np.logical_or(filled_mask, hole)

            return filled_mask.astype(np.uint8)

        except Exception as e:
            print(f"Error in fill_holes: {str(e)}")
            return binary_mask.astype(np.uint8)

    # Save the combined mask
    if combined_mask is not None:
        if bypass_filling:
            #print("Bypassing the hole-filling step as 'bypass_filling' is set to True.")
            combined_mask_filled = combined_mask
        else:
            #print("Applying the hole-filling step.")
            combined_mask_filled = fill_holes(combined_mask)

        combined_mask_grayscale = (combined_mask_filled.astype(np.uint8) * 255)
        combined_mask_image = Image.fromarray(combined_mask_grayscale)
        combined_output_path = os.path.join(
            output_folder,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_combined_mask_{text_prompt}.png"
        )
        combined_mask_image.save(combined_output_path)
        #print(f"Saved combined mask to {combined_output_path}")
    else:
        print(f"No masks to combine for {image_path}.")

    print("=== Debug: Exiting extractImages ===\n")

def extractImages(boxes_xyxy, image_path: str, text_prompt: str, output_folder: str, bypass_filling=False):
    #This uses the old SAM which actually works better

    if len(boxes_xyxy) == 0:
        print(f"No bounding boxes provided for {image_path}. Skipping.")
        return
    
    if mode not in ["predict", "segment"]:
        print(f"Invalid mode: {mode}. Supported modes are 'predict' and 'segment'.")
        return

    # 2. Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 3. Create a SAMPredictor with some global overrides
    #    Adjust or add as needed: conf, imgsz, model, etc.
    overrides = dict(
        conf=0.4,         # Confidence threshold
        mode=mode,   # Ensures we do inference
        imgsz=1024,       # Higher resolution may yield finer masks
        model="sam_l.pt"  # If needed, or use a different checkpoint
    )
    predictor = SAMPredictor(overrides=overrides)

    try:
        # 4. Load the image into the predictor
        predictor.set_image(image_path)

        # 5. Call predictor based on mode
        if mode == "predict":
            print(f"Running SAM in 'predict' mode on {image_path}...")
            results = predictor(
                bboxes=boxes_xyxy,
                points_stride=64,  # Advanced argument
                crop_n_layers=1,  # Advanced argument
            )

        elif mode == "segment":
            print(f"Running SAM in 'segment' mode on {image_path}...")
            results = predictor()

        print(f"Results type: {type(results)}")
        if isinstance(results, list):
            print(f"Number of result objects: {len(results)}")
            for idx, r in enumerate(results):
                print(f"Result {idx + 1}: Masks available: {hasattr(r, 'masks')}")

    except Exception as e:
        print(f"Error running SAM on {image_path}: {e}")
        sys.exit(1)

    # Helper function: merges masks from a single Results object into combined_mask
    def combine_masks_from_results(results_obj, combined_mask):
        """Merge all mask tensors from one Results object into combined_mask."""
        if results_obj.masks is not None and hasattr(results_obj.masks, "data"):
            for mask_tensor in results_obj.masks.data:
                mask_numpy = mask_tensor.cpu().numpy()
                if combined_mask is None:
                    combined_mask = mask_numpy
                else:
                    combined_mask = np.logical_or(combined_mask, mask_numpy)
        else:
            print(f"No masks detected for {image_path}.")
        return combined_mask

    # 6. Merge all masks into a single combined_mask
    combined_mask = None
    if isinstance(results, list):
        # We got multiple Results objects
        for r in results:
            combined_mask = combine_masks_from_results(r, combined_mask)
    else:
        # We got a single Results object
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
            filled_mask = binary_mask.copy()
            for i in range(1, num_features + 1):
                hole = (labeled_holes == i)
                # If the hole doesn't touch the boundary, fill it
                if not (hole[0, :].any() or hole[-1, :].any() or hole[:, 0].any() or hole[:, -1].any()):
                    filled_mask = np.logical_or(filled_mask, hole)
            return filled_mask.astype(np.uint8)
        except Exception as e:
            print(f"Error in fill_holes: {str(e)}")
            return binary_mask.astype(np.uint8)

    # 8. Save the combined mask
    if combined_mask is not None:
        combined_mask_filled = combined_mask if bypass_filling else fill_holes(combined_mask)
        combined_mask_grayscale = (combined_mask_filled.astype(np.uint8) * 255)
        combined_mask_image = Image.fromarray(combined_mask_grayscale)
        combined_output_path = os.path.join(
            output_folder,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_combined_mask_{text_prompt}.png"
        )
        combined_mask_image.save(combined_output_path)
    else:
        print(f"No masks to combine for {image_path}.")

    print("=== Debug: Exiting extractImages ===\n")

def worker_process_image(args):
    image_path, text_prompt, box_threshold, output_folder = args
    try:
        boxes_xyxy, annotated_frame = createBoxes(image_path, text_prompt, box_threshold)
        extractImages(boxes_xyxy, image_path, text_prompt, output_folder)
        logging.info(f"Processed {image_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
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

def process_images(root_folder, output_folder, start_from_zero=True, selected_tags=None, log_callback=None):
    """
    Processes images in the input folder based on selected tags and saves them to the output folder.
    """

    if not os.path.exists(global_folder):
        raise FileNotFoundError(f"Global folder not found: {global_folder}")

    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    
    if log_callback is None:
        log_callback = print
    
    if selected_tags is None:
        selected_tags = {}

    log_file = f'{global_folder}/process_log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO)

    # If start_from_zero is True, erase the log file
    if start_from_zero:
        open(log_file, 'w').close()
        last_processed_image = None
    else:
        last_processed_image = get_last_processed_image(log_file)

    log_callback("Starting image processing...")
    log_callback(f"Root folder: {root_folder}")
    log_callback(f"Output folder: {output_folder}")
    log_callback(f"Selected tags: {selected_tags}")
    log_callback(f"Start from zero: {start_from_zero}")

    # Build the list of tasks
    tasks = []
    for text_prompt, box_threshold in selected_tags.items():
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

    log_callback(f"Total tasks to process: {len(tasks)}")

    # Process tasks with multiprocessing
    with Pool(processes=1, initializer=worker_init) as pool:
        results = pool.imap_unordered(worker_process_image, tasks)
        with tqdm(total=len(tasks), desc="Processing Images") as pbar:
            for result in results:
                if result:  # Log successful processing
                    log_callback("Task completed successfully.")
                else:  # Log any failures
                    log_callback("Task failed.")
                pbar.update(1)

    # Ensure the log file exists if it didn't before
    if last_processed_image is None:
        open(log_file, 'a').close()

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("Your current device is:", device)
    root_folder = f'/Users/tommasoprinetti/Desktop/Upscaled_Images'
    output_folder = f'{global_folder}/OUTPUT_MASKS'
    process_images(root_folder, output_folder, start_from_zero=True)