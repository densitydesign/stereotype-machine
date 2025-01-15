"""
Image Cutting Module

This package contains functionalities for:
1. Bounding box creation using GroundingDINO.
2. Mask refinement and extraction using SAM (Segment Anything Model).
3. Iterative processing of image directories.

Modules:
- createBoxes: Generate bounding boxes for objects in an image based on text prompts.
- extractImages: Refine masks and save them as black-and-white mask images.
- process_images: Process multiple images in a folder using predefined text prompts.

Dependencies:
- GroundingDINO for bounding box generation.
- SAM for mask refinement.
- PyTorch, OpenCV, and other utilities.

Author: Tommaso Prinetti
"""

from .image_cutting import createBoxes, extractImages, process_images

__all__ = ["createBoxes", "extractImages", "process_images"]
