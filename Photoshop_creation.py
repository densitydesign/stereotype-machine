#Photoshop_creation.py
import os
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from typing import List, Tuple, Optional

# Try to import psd_tools, install if not available
try:
    from psd_tools import PSDImage
    from psd_tools.constants import ColorMode, BlendMode
except ImportError:
    print("Installing psd_tools library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psd-tools"])
    from psd_tools import PSDImage
    from psd_tools.constants import ColorMode, BlendMode

# Get the script's root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

# Create photoshop_files directory if it doesn't exist
photoshop_files_dir = os.path.join(project_root, "photoshop_files")
os.makedirs(photoshop_files_dir, exist_ok=True)

# ============ CONFIGURATION ============
# Set your paths here - use relative paths from the script's location
CONFIG = {
    # Path to your grid image (relative to script location)
    "grid_image_path": "path/to/your/grid_image.png",
    
    # Path to the folder containing your masks (relative to script location)
    "masks_folder": "path/to/your/masks_folder",
}

# Automatically set the output path based on the grid image name
def get_output_path(grid_image_path: str) -> str:
    """Generate output path in photoshop_files directory using grid image name."""
    grid_filename = os.path.basename(grid_image_path)
    grid_name = os.path.splitext(grid_filename)[0]
    return os.path.join(photoshop_files_dir, f"{grid_name}.psd")

# =====================================

def create_psd_with_grid_and_masks(
    grid_image_path: str,
    masks_folder: str,
    output_path: str,
    verbose: bool = True
) -> str:
    """
    Create a Photoshop PSD file with a grid image as background and masks as layer masks.
    
    Args:
        grid_image_path: Path to the grid image to use as background
        masks_folder: Folder containing mask images to apply as layer masks
        output_path: Path where the PSD file will be saved
        verbose: Whether to print progress information
        
    Returns:
        Path to the created PSD file
    """
    if verbose:
        print(f"Creating PSD file with grid image and masks...")
    
    # Load the grid image
    try:
        grid_img = Image.open(grid_image_path)
        if verbose:
            print(f"Loaded grid image: {grid_image_path} ({grid_img.width}x{grid_img.height})")
    except Exception as e:
        raise ValueError(f"Error loading grid image {grid_image_path}: {e}")
    
    # Get all mask images from the folder
    mask_files = [
        f for f in os.listdir(masks_folder)
        if os.path.isfile(os.path.join(masks_folder, f)) and 
        f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))
    ]
    
    # Sort to ensure consistent order
    mask_files.sort()
    
    if not mask_files:
        raise ValueError(f"No mask images found in folder: {masks_folder}")
    
    if verbose:
        print(f"Found {len(mask_files)} mask images in {masks_folder}")
    
    # Create a new PSD file with the dimensions of the grid image
    psd = PSDImage.new(width=grid_img.width, height=grid_img.height, color_mode=ColorMode.RGB)
    
    # Add the grid image as the background layer
    bg_layer = psd.create_layer(name="Background")
    
    # Convert grid image to RGBA if it's not already
    if grid_img.mode != 'RGBA':
        grid_img = grid_img.convert('RGBA')
    
    # Set the background layer pixels
    bg_layer.pixels = np.array(grid_img)
    
    # Load and add each mask as a layer with layer mask
    for mask_file in tqdm(mask_files, desc="Adding masks", disable=not verbose):
        mask_path = os.path.join(masks_folder, mask_file)
        mask_name = os.path.splitext(mask_file)[0]
        
        try:
            # Load the mask image
            mask_img = Image.open(mask_path)
            
            # Create a new layer with the same content as the grid image
            layer = psd.create_layer(name=f"Layer {mask_name}")
            
            # Set the layer pixels to be the same as the grid image
            layer.pixels = np.array(grid_img)
            
            # Convert mask to grayscale if it's not already
            if mask_img.mode != 'L':
                mask_img = mask_img.convert('L')
            
            # Resize mask if needed to match the grid image dimensions
            if mask_img.width != grid_img.width or mask_img.height != grid_img.height:
                if verbose:
                    print(f"Resizing mask {mask_file} to match grid image dimensions")
                mask_img = mask_img.resize((grid_img.width, grid_img.height), Image.LANCZOS)
            
            # Set the layer mask
            layer.mask = np.array(mask_img)
            
            if verbose:
                print(f"Added layer with mask: {mask_name}")
                
        except Exception as e:
            print(f"Error processing mask {mask_file}: {e}")
    
    # Save the PSD file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    psd.save(output_path)
    
    if verbose:
        print(f"PSD file created successfully: {output_path}")
    
    return output_path

def create_psd_with_layers(
    output_path: str, 
    image_paths: List[str], 
    layer_names: Optional[List[str]] = None,
    canvas_size: Optional[Tuple[int, int]] = None,
    verbose: bool = True
) -> str:
    """
    Create a Photoshop PSD file with multiple layers from the provided images.
    
    Args:
        output_path: Path where the PSD file will be saved
        image_paths: List of paths to images to be used as layers
        layer_names: Optional list of names for each layer (defaults to filenames if not provided)
        canvas_size: Optional tuple (width, height) for the canvas size
                    (defaults to the size of the largest image)
        verbose: Whether to print progress information
        
    Returns:
        Path to the created PSD file
    """
    if verbose:
        print(f"Creating PSD file with {len(image_paths)} layers...")
    
    # Validate inputs
    if not image_paths:
        raise ValueError("No image paths provided")
    
    if layer_names and len(layer_names) != len(image_paths):
        raise ValueError(f"Number of layer names ({len(layer_names)}) doesn't match number of images ({len(image_paths)})")
    
    # Load all images and determine canvas size if not specified
    images = []
    max_width, max_height = 0, 0
    
    for img_path in tqdm(image_paths, desc="Loading images", disable=not verbose):
        try:
            img = Image.open(img_path)
            images.append(img)
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a placeholder red image for errors
            img = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
            images.append(img)
    
    # Use provided canvas size or default to the largest image dimensions
    width, height = canvas_size if canvas_size else (max_width, max_height)
    
    # Create a new PSD file
    psd = PSDImage.new(width=width, height=height, color_mode=ColorMode.RGB)
    
    # Generate default layer names if not provided
    if not layer_names:
        layer_names = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
    
    # Add layers in reverse order (bottom to top in Photoshop)
    for i, (img, name) in enumerate(zip(reversed(images), reversed(layer_names))):
        if verbose:
            print(f"Adding layer: {name}")
        
        # Convert image to RGBA if it's not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Center the image on the canvas
        paste_x = (width - img.width) // 2
        paste_y = (height - img.height) // 2
        
        # Create a new layer
        layer = psd.create_layer(name=name)
        
        # Convert PIL Image to numpy array and set as layer pixels
        layer_array = np.array(img)
        layer.pixels = layer_array
        
        # Set layer position
        layer.offset = (paste_x, paste_y)
    
    # Save the PSD file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    psd.save(output_path)
    
    if verbose:
        print(f"PSD file created successfully: {output_path}")
    
    return output_path

def create_psd_from_folders(
    original_folder: str,
    mask_folder: str,
    output_path: str,
    mask_suffix: str = "_combined_mask.png",
    verbose: bool = True
) -> str:
    """
    Create a PSD file with matching original images and their masks as separate layers.
    
    Args:
        original_folder: Folder containing original images
        mask_folder: Folder containing mask images
        output_path: Path where the PSD file will be saved
        mask_suffix: Suffix used in mask filenames
        verbose: Whether to print progress information
        
    Returns:
        Path to the created PSD file
    """
    if verbose:
        print(f"Creating PSD from folders:\nOriginals: {original_folder}\nMasks: {mask_folder}")
    
    # Get all original images
    original_images = [
        os.path.join(original_folder, f) for f in os.listdir(original_folder)
        if os.path.isfile(os.path.join(original_folder, f)) and 
        f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))
    ]
    
    # Sort to ensure consistent order
    original_images.sort()
    
    if not original_images:
        raise ValueError(f"No images found in original folder: {original_folder}")
    
    # For each original image, find its corresponding mask
    image_pairs = []
    layer_names = []
    
    for orig_path in original_images:
        orig_filename = os.path.basename(orig_path)
        orig_name = os.path.splitext(orig_filename)[0]
        
        # Determine mask filename
        mask_filename = f"{orig_name}{mask_suffix}"
        mask_path = os.path.join(mask_folder, mask_filename)
        
        if os.path.exists(mask_path):
            image_pairs.extend([orig_path, mask_path])
            layer_names.extend([f"{orig_name}_original", f"{orig_name}_mask"])
        else:
            if verbose:
                print(f"Warning: No mask found for {orig_filename}")
            image_pairs.append(orig_path)
            layer_names.append(f"{orig_name}_original")
    
    if not image_pairs:
        raise ValueError("No valid image pairs found")
    
    # Create the PSD file
    return create_psd_with_layers(
        output_path=output_path,
        image_paths=image_pairs,
        layer_names=layer_names,
        verbose=verbose
    )

def create_psd_from_grid_folders(
    original_grid_folder: str,
    mask_grid_folder: str,
    output_folder: str,
    verbose: bool = True
) -> List[str]:
    """
    Create PSD files for each pair of matching grid images in the original and mask folders.
    
    Args:
        original_grid_folder: Folder containing original grid images
        mask_grid_folder: Folder containing mask grid images
        output_folder: Folder where PSD files will be saved
        verbose: Whether to print progress information
        
    Returns:
        List of paths to created PSD files
    """
    if verbose:
        print(f"Creating PSDs from grid folders:\nOriginals: {original_grid_folder}\nMasks: {mask_grid_folder}")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all original grid images
    original_grids = [
        f for f in os.listdir(original_grid_folder)
        if os.path.isfile(os.path.join(original_grid_folder, f)) and 
        f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))
    ]
    
    # Sort to ensure consistent order
    original_grids.sort()
    
    if not original_grids:
        raise ValueError(f"No grid images found in original folder: {original_grid_folder}")
    
    created_psds = []
    
    for orig_grid in tqdm(original_grids, desc="Creating PSD files", disable=not verbose):
        orig_name = os.path.splitext(orig_grid)[0]
        
        # Look for matching mask grid with same name
        mask_candidates = [
            f for f in os.listdir(mask_grid_folder)
            if os.path.splitext(f)[0] == orig_name
        ]
        
        if not mask_candidates:
            if verbose:
                print(f"Warning: No matching mask grid found for {orig_grid}")
            continue
        
        mask_grid = mask_candidates[0]
        
        # Create paths
        orig_path = os.path.join(original_grid_folder, orig_grid)
        mask_path = os.path.join(mask_grid_folder, mask_grid)
        output_path = os.path.join(output_folder, f"{orig_name}.psd")
        
        # Create PSD with these two layers
        try:
            create_psd_with_layers(
                output_path=output_path,
                image_paths=[orig_path, mask_path],
                layer_names=["Original Grid", "Mask Grid"],
                verbose=False
            )
            created_psds.append(output_path)
            if verbose:
                print(f"Created: {output_path}")
        except Exception as e:
            print(f"Error creating PSD for {orig_grid}: {e}")
    
    if verbose:
        print(f"Created {len(created_psds)} PSD files in {output_folder}")
    
    return created_psds

def main():
    """Main function to run the script directly."""
    try:
        # Convert relative paths to absolute paths
        grid_image_abs_path = os.path.join(project_root, CONFIG["grid_image_path"])
        masks_folder_abs_path = os.path.join(project_root, CONFIG["masks_folder"])
        
        # Get output path based on grid image name
        output_path = get_output_path(grid_image_abs_path)
        
        create_psd_with_grid_and_masks(
            grid_image_path=grid_image_abs_path,
            masks_folder=masks_folder_abs_path,
            output_path=output_path
        )
        print(f"✅ PSD file created successfully: {output_path}")
    except Exception as e:
        print(f"❌ Error creating PSD file: {e}")

if __name__ == "__main__":
    main()
