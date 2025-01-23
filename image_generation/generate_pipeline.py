#generate_pipeline.py

import os
import requests
import base64
import io
import time
import psutil
import platform
import subprocess
from tqdm import tqdm
from PIL import Image
import sys

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory and config directory to sys.path
sys.path.append(current_dir)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import modules from the config folder
from api_parameters_txt2img import txt2img_data
from api_parameters_img2img import img2img_data

# Base folders for image saving
BASE_FOLDER = "Generated_Images"
UPSCALED_FOLDER = "Upscaled_Images"

def is_drawthings_running():
    """
    Checks if the DrawThings app is running.
    Returns:
        bool: True if the app is running, False otherwise.
    """
    for process in psutil.process_iter(['name']):
        if 'DrawThings' in process.info['name']:  # Adjust the app name based on actual process name
            return True
    return False

def start_drawthings():
    """
    Starts the DrawThings app if not running.
    Returns:
        None
    """
    system_platform = platform.system()
    try:
        if system_platform == "Darwin":  # macOS
            subprocess.Popen(["open", "/Applications/Draw Things.app"])
        elif system_platform == "Windows":
            subprocess.Popen(["start", "path_to_DrawThings.exe"], shell=True)
        elif system_platform == "Linux":
            subprocess.Popen(["xdg-open", "/path/to/DrawThings"])  # Update with the actual path
        else:
            print("Unsupported platform. Cannot start DrawThings app.")
        time.sleep(5)  # Allow some time for the app to initialize
    except Exception as e:
        print(f"Error starting DrawThings app: {e}")

def generate_images(nation, category, num_images, generated_folder=None, upscaled_folder=None, steps=None):
    """
    Generates images using the txt2img API based on specified nation, category, and number of images.
    Args:
        nation (str): The nation for which images will be generated.
        category (str): The category of images (e.g., "family" or "working").
        num_images (int): The number of images to generate.
        generated_folder (str): The folder where generated images will be saved.
        upscaled_folder (str): The folder where upscaled images will be saved.
        steps (int): The number of steps for the txt2img generation process.
    Returns:
        str: The path of the last generated image (or None if no images were generated).
    """

    if not is_drawthings_running():
        print("DrawThings app is not running. Starting it now...")
        start_drawthings()
    else:
        print("DrawThings app is already running.")

    # Use provided folder paths if specified; otherwise, use default
    generated_folder = generated_folder or BASE_FOLDER
    upscaled_folder = upscaled_folder or UPSCALED_FOLDER

    # Define the folder path for the current nation and category
    nation_folder = os.path.join(generated_folder, nation)
    category_folder = os.path.join(nation_folder, category)
    os.makedirs(category_folder, exist_ok=True)

    # Set the steps for image generation
    txt2img_data["steps"] = steps

    # Construct the prompt directly from GUI inputs
    prompt = f"{nation} {category}, (35mm lens photography), extremely detailed, 4k, shot on dslr, photorealistic, photographic, sharp"
    txt2img_data["prompt"] = prompt

    last_image_path = None  

    # Generate the specified number of images
    for i in tqdm(range(num_images), desc=f"Generating images for {nation} - {category}"):
        try:
            # Send a POST request to the txt2img API with the updated parameters
            response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=txt2img_data)
            
            # Check for successful response
            if response.status_code != 200:
                print(f"Failed to generate image {i + 1}: {response.text}")
                continue

            # Parse the JSON response
            r = response.json()
            
            # Loop through the images in the response
            for idx_img, image_base64 in enumerate(r['images']):
                # Decode the base64 image data
                image_data = base64.b64decode(image_base64.split(",", 1)[-1])
                image = Image.open(io.BytesIO(image_data))

                # Define a unique filename for the image
                image_filename = f"{nation}_{category}_{i + 1}_{idx_img + 1}.png"
                image_path = os.path.join(category_folder, image_filename)
                while os.path.exists(image_path):
                    idx_img += 1  # Increment to get a unique file name
                    image_filename = f"{nation}_{category}_{i + 1}_{idx_img + 1}.png"
                    image_path = os.path.join(category_folder, image_filename)

                # Save the image as a PNG file
                image.save(image_path)
                print(f"Image saved at {image_path}")

                # Proceed to img2img upscale
                last_image_path = upscale_image(image_path, nation, category, upscaled_folder)

        except Exception as e:
            print(f"Error while generating images for {nation} - {category}: {e}")

    return last_image_path
def upscale_image(image_path, nation, category, upscaled_folder):
    """
    Upscales an image using the img2img API and saves the upscaled image.
    Args:
        image_path (str): Path to the image to be upscaled.
        nation (str): The nation for which images were generated.
        category (str): The category of images (e.g., "family" or "working").
        upscaled_folder (str): The folder where upscaled images will be saved.
    Returns:
        None
    """
    # Convert the image to a base64 string
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        print(f"Failed to encode image for img2img: {image_path}")
        return

    # Update img2img_data with the base64 image
    img2img_data["init_images"] = [base64_image]

    # Validate img2img_data before sending the request
    error_list = validate_img2img_data(img2img_data)
    if error_list:
        print("Validation failed for img2img data:")
        for error in error_list:
            print(error)
        return

    try:
        # Send a POST request to the img2img API with the updated parameters
        response = requests.post("http://127.0.0.1:7860/sdapi/v1/img2img", json=img2img_data)

        # Check for successful response
        if response.status_code != 200:
            print(f"Failed to upscale image: {response.text}")
            return

        # Parse the JSON response
        r = response.json()

        # Define folder path for saving upscaled images
        final_nation_folder = os.path.join(upscaled_folder, nation)
        final_category_folder = os.path.join(final_nation_folder, category)
        os.makedirs(final_category_folder, exist_ok=True)

        # Loop through each image in the response
        for idx, final_image_base64 in enumerate(r['images']):
            # Decode the base64 image data
            final_image_data = base64.b64decode(final_image_base64.split(",", 1)[-1])
            final_image = Image.open(io.BytesIO(final_image_data))

            # Define a unique filename and path for the upscaled image
            final_image_filename = os.path.basename(image_path).replace(".png", f"_upscaled_{idx + 1}.png")
            final_image_path = os.path.join(final_category_folder, final_image_filename)
            while os.path.exists(final_image_path):
                idx += 1  # Increment index to avoid overwriting
                final_image_filename = os.path.basename(image_path).replace(".png", f"_upscaled_{idx + 1}.png")
                final_image_path = os.path.join(final_category_folder, final_image_filename)

            # Save the upscaled image
            final_image.save(final_image_path)
            print(f"Upscaled image saved at {final_image_path}")

    except Exception as e:
        print(f"Error during img2img upscaling for image {image_path}: {e}")

    return final_image_path

def encode_image_to_base64(image_path):
    """
    Encodes an image at the specified path to a base64 string.

    Args:
        image_path (str): Path to the image to encode.

    Returns:
        str: Base64-encoded string of the image, or None if encoding fails.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
            base64_image = base64.b64encode(image_content).decode()
            return base64_image
    except FileNotFoundError:
        print(f"Error: Image file {image_path} not found.")
        return None

def validate_img2img_data(data):
    """
    Validates the img2img data dictionary to ensure all required fields are present and correctly formatted.

    Args:
        data (dict): The img2img data to validate.

    Returns:
        list: List of error messages if validation fails; empty if data is valid.
    """
    errors = []
    # Required fields and their expected types
    required_fields = {
        "negative_prompt_for_image_prior": bool,
        "motion_scale": (int, float),
        "fps": (int, float),
        "guidance_scale": (int, float),
        "steps": int,
        "controls": list,
        "strength": (int, float),
        "init_images": list,
        "height": int,
        "width": int,
        "prompt": str,
        "negative_prompt": str,
        "model": str,
    }

    # Check for missing required fields
    for field, field_type in required_fields.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
        else:
            if not isinstance(data[field], field_type):
                errors.append(f"Field '{field}' must be of type {field_type.__name__}, got {type(data[field]).__name__}")

    # Validate 'controls' list
    if "controls" in data:
        if not isinstance(data["controls"], list):
            errors.append("Field 'controls' must be a list")
        else:
            for idx, control in enumerate(data["controls"]):
                if not isinstance(control, dict):
                    errors.append(f"Control at index {idx} must be a dictionary")
                    continue
                # Required fields in each control dictionary
                control_required_fields = {"inputOverride": str, "targetBlocks": list, "file": str}
                for ctrl_field, ctrl_type in control_required_fields.items():
                    if ctrl_field not in control:
                        errors.append(f"Missing required field '{ctrl_field}' in control at index {idx}")
                    elif not isinstance(control[ctrl_field], ctrl_type):
                        errors.append(f"Field '{ctrl_field}' in control at index {idx} must be of type {ctrl_type.__name__}")
    
    return errors
