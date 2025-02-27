#create_grids.py
import os
from PIL import Image
from tqdm import tqdm
import gc
import subprocess
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

# Default grid gap
grid_gap = 100

def find_leaf_folders(root_dir):
    """
    Find all leaf folders (folders with no subfolders) in the directory tree.
    Returns a list of tuples (folder_path, folder_name).
    """
    print("\n=== Debug: Finding leaf folders ===")
    leaf_folders = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # If this directory has no subdirectories, it's a leaf
        if not dirnames:
            # Get image files in this folder
            image_files = [f for f in filenames if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
            if image_files:  # Only include folders that contain images
                folder_name = os.path.basename(dirpath)
                parent_path = os.path.dirname(dirpath)
                parent_name = os.path.basename(parent_path)
                grandparent_path = os.path.dirname(parent_path)
                grandparent_name = os.path.basename(grandparent_path)
                
                # Create a path that preserves the structure: nationality/category/type
                relative_path = os.path.join(grandparent_name, parent_name, folder_name)
                print(f"Found leaf folder: {dirpath} with {len(image_files)} images")
                leaf_folders.append((dirpath, relative_path))
    
    print(f"Total leaf folders found: {len(leaf_folders)}")
    return leaf_folders

def compress_png(input_path):
    """
    Compress PNG using pngquant for best compression with minimal quality loss.
    Falls back to PIL's built-in compression if pngquant is not available.
    """
    try:
        # Get original file size
        original_size = os.path.getsize(input_path)
        
        # Use pngquant (best compression with minimal quality loss)
        temp_output = input_path + ".temp.png"
        subprocess.run(["pngquant", "--force", "--output", temp_output, "--quality=85-95", input_path], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.move(temp_output, input_path)
        
        # Get compressed file size
        compressed_size = os.path.getsize(input_path)
        
        # Calculate and print savings
        size_diff = original_size - compressed_size
        percent_saved = (size_diff / original_size) * 100 if original_size > 0 else 0
        print(f"Compressed {input_path} with pngquant: {original_size/1024:.2f}KB → {compressed_size/1024:.2f}KB (saved {percent_saved:.2f}%)")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("pngquant not available, using PIL's built-in compression")
        return False

def create_image_grids_from_structure(root_dir, grid_size, output_dir=None):
    """
    Create image grids for each leaf folder in the directory structure.
    """
    print("\n=== Debug: Creating image grids from folder structure ===")
    os.makedirs(output_dir, exist_ok=True)

    # Find all leaf folders
    leaf_folders = find_leaf_folders(root_dir)
    
    for folder_path, relative_path in leaf_folders:
        # Get all image files in the folder
        image_files = [f for f in os.listdir(folder_path) 
                      if os.path.isfile(os.path.join(folder_path, f)) and 
                      f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
        
        if not image_files:
            print(f"No images found in {folder_path}, skipping...")
            continue
        
        # Sort the image files to maintain consistent order
        image_files.sort()
            
        print(f"Processing folder: {folder_path} with {len(image_files)} images")
        image_paths = [os.path.join(folder_path, f) for f in image_files]
        
        # Process images in batches to create grids
        for i in tqdm(range(0, len(image_paths), grid_size[0] * grid_size[1]), 
                     desc=f"Processing {relative_path}"):
            grid_images = image_paths[i:i + grid_size[0] * grid_size[1]]

            # Load images
            images = []
            for img_path in tqdm(grid_images, desc="Loading images", leave=False):
                try:
                    img = Image.open(img_path).convert('RGBA')
                    images.append(img)
                    print(f"Loaded image: {img_path}")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

            if not images:
                print(f"No valid images found for grid starting at index {i}")
                continue

            # Calculate grid dimensions
            max_width = max(img.width for img in images)
            max_height = max(img.height for img in images)
            gap = grid_gap
            grid_width = max_width * grid_size[0] + gap * (grid_size[0] - 1)
            grid_height = max_height * grid_size[1] + gap * (grid_size[1] - 1)
            print(f"Grid dimensions: {grid_width}x{grid_height}")

            # Create a blank grid image
            grid_img = Image.new('RGBA', (grid_width, grid_height), (0, 0, 0, 0))

            # Paste images into the grid
            for idx, img in enumerate(images):
                row = idx // grid_size[0]
                col = idx % grid_size[0]
                x_position = col * (max_width + gap)
                y_position = row * (max_height + gap)
                grid_img.paste(img, (x_position, y_position))
                print(f"Pasted image at position ({x_position}, {y_position})")

            # Create output directory structure
            grid_subdir = os.path.join(output_dir, relative_path)
            print(f"Debug: Preparing to save grid in directory: {grid_subdir}")
            os.makedirs(grid_subdir, exist_ok=True)

            # Extract components from relative_path for naming
            path_parts = relative_path.split(os.path.sep)
            if len(path_parts) >= 3:
                nationality, category, image_type = path_parts[-3], path_parts[-2], path_parts[-1]
                grid_name = f"{nationality}_{category}_{image_type}_grid_{i // (grid_size[0] * grid_size[1]) + 1}.png"
            else:
                grid_name = f"grid_{i // (grid_size[0] * grid_size[1]) + 1}.png"
                
            grid_output_path = os.path.join(grid_subdir, grid_name)
            
            try:
                print(f"Debug: Attempting to save grid at {grid_output_path}")
                # Convert to RGB if saving with compression (PNG with alpha doesn't support quality)
                if grid_img.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', grid_img.size, (255, 255, 255))
                    # Paste the image with alpha on the background
                    background.paste(grid_img, (0, 0), grid_img)
                    # Save with PIL's basic compression
                    background.save(grid_output_path, optimize=True)
                else:
                    # Save with PIL's basic compression
                    grid_img.save(grid_output_path, optimize=True)
                
                # Apply pngquant compression
                compress_png(grid_output_path)
                print(f"Saved grid image: {grid_output_path}")
            except Exception as e:
                print(f"Error saving grid image {grid_output_path}: {e}")
            
            # Clean up to free memory
            for img in images:
                img.close()
            grid_img.close()
            images.clear()
            gc.collect()

    print("\n=== Debug: Completed creating image grids ===")
    # Final memory cleanup
    gc.collect()