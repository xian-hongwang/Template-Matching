# Template Matching Using Sliding Window and LoFTR

This project implements a template matching technique using two different approaches:
1. **Sliding Window with Multiple Feature Extractors**  
2. **LoFTR (Local Feature Transformer)**

The goal of this project is to identify the best match of a small template image within a larger reference image. This task is essential in various computer vision applications such as object detection, image registration, and automated inspection.

## Project Structure

The repository includes multiple files:
- **inference_sliding_multiscaleV3.py**: Implements template matching using a sliding window approach with multiple feature extractors (DINOv2, ResNet, Swin Transformer, and MobileNet).
- **run_sliding_window.py**: Script to execute the sliding window template matching.
- **run_LoFTR.py**: Script that uses the LoFTR model for feature matching between a template image and a larger reference image.

## Requirements

To run the project, ensure you have the following dependencies installed:
```bash
pip install torch torchvision matplotlib kornia opencv-python
