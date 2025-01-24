#create_grids.py
import os
from PIL import Image
from tqdm import tqdm
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

# Input and output directories
root_dir = ''
output_dir = ''
grid_gap = 100

def find_images(root_dir,nation_pattern, category_pattern):
    """
    Find and group images based on nation and category (e.g., family, working).
    """
    print("\n=== Debug: Starting image search ===")
    nation_category_dict = {}
    
    for subdir, _, files in os.walk(root_dir):
        print(f"Scanning directory: {subdir}")
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                nation_match = re.search(nation_pattern, file, re.IGNORECASE)
                category_match = re.search(category_pattern, file, re.IGNORECASE)
                
                if nation_match and category_match:
                    nation = nation_match.group(0).capitalize()
                    category = category_match.group(0).lower()
                    print(f"Matched file: {file} -> Nation: {nation}, Category: {category}")

                    # Add to dictionary
                    if nation not in nation_category_dict:
                        nation_category_dict[nation] = {}
                    if category not in nation_category_dict[nation]:
                        nation_category_dict[nation][category] = []
                    
                    image_path = os.path.join(subdir, file)
                    nation_category_dict[nation][category].append(image_path)

    print("\n=== Debug: Completed image search ===")
    return nation_category_dict

def create_image_grids(root_dir, category_pattern, grid_size=(6, 6), output_dir=None, nation_pattern=None):
    """
    Create image grids for each nation and category.
    """
    print("\n=== Debug: Creating image grids ===")
    os.makedirs(output_dir, exist_ok=True)

    # Find images grouped by nation and category
    nation_category_dict = find_images(root_dir, nation_pattern,category_pattern)
    for nation, categories in nation_category_dict.items():
        for category, image_paths in categories.items():

            print(f"Found {len(image_paths)} images for nation '{nation}', category '{category}'")

            # Process images in batches to create grids
            for i in tqdm(range(0, len(image_paths), grid_size[0] * grid_size[1]), desc=f"Processing {nation} - {category}"):
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

                # Save the grid image
                grid_subdir = os.path.join(output_dir, nation, category)
                print(f"Debug: Preparing to save grid in directory: {grid_subdir}")
                os.makedirs(grid_subdir, exist_ok=True)

                grid_output_path = os.path.join(grid_subdir, f"{nation}_{category}_grid_{i // (grid_size[0] * grid_size[1]) + 1}.png")
                try:
                    print(f"Debug: Attempting to save grid at {grid_output_path}")
                    grid_img.save(grid_output_path)
                    print(f"Saved grid image: {grid_output_path}")
                except Exception as e:
                    print(f"Error saving grid image {grid_output_path}: {e}")

    print("\n=== Debug: Completed creating image grids ===")