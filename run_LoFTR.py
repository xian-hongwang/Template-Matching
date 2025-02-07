import torch
import cv2
import numpy as np
from kornia.feature import LoFTR
import matplotlib.pyplot as plt

# Define file paths
TEMPLATE_IMAGE_PATH = "./template.bmp"  # Small image
LARGE_IMAGE_PATH = "./target.bmp"  # Large reference image
OUTPUT_IMAGE_PATH = "matched_result.png"  # Large image with bounding box
MATCH_VISUALIZATION_PATH = "match_visualization.png"  # Side-by-side matching visualization

# Enhance contrast using CLAHE (Adaptive Histogram Equalization)
def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(img)

# Convert images to tensors for LoFTR input
def preprocess_image(img):
    img = torch.tensor(img, dtype=torch.float32) / 255.0  # Normalize to [0,1]
    img = img.unsqueeze(0).unsqueeze(0)  # Add batch & channel dimensions
    return img

def match_template():
    """Perform feature matching using LoFTR and visualize the detected region."""
    
    # Load images
    template_img = cv2.imread(TEMPLATE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)  # Small image
    large_img_color = cv2.imread(LARGE_IMAGE_PATH, cv2.IMREAD_COLOR)  # Large image (color)
    
    if template_img is None or large_img_color is None:
        print("Error: Unable to load one or both images. Check file paths.")
        return

    large_img = cv2.cvtColor(large_img_color, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for LoFTR

    # Enhance contrast
    template_img = enhance_contrast(template_img)
    large_img = enhance_contrast(large_img)

    # Convert images to tensors
    template_tensor = preprocess_image(template_img)
    large_tensor = preprocess_image(large_img)

    # Initialize LoFTR model
    loftr = LoFTR(pretrained="outdoor")
    loftr.eval()

    # Perform feature matching using LoFTR
    with torch.no_grad():
        input_dict = {"image0": template_tensor, "image1": large_tensor}
        correspondences = loftr(input_dict)

    # Extract matched keypoints from both images
    mkpts1 = correspondences["keypoints0"].cpu().numpy()  # Keypoints from the small image
    mkpts2 = correspondences["keypoints1"].cpu().numpy()  # Corresponding keypoints in the large image

    # If enough keypoints are found, compute the Homography
    if len(mkpts1) >= 4:
        H, mask = cv2.findHomography(mkpts1, mkpts2, cv2.RANSAC, 5.0)

        # Define the bounding box of the small image
        h_template, w_template = template_img.shape
        template_corners = np.array([[0, 0], [w_template, 0], [w_template, h_template], [0, h_template]], dtype=np.float32).reshape(-1, 1, 2)

        # Transform template corners to large image coordinates
        mapped_corners = cv2.perspectiveTransform(template_corners, H).astype(int)

        # === 🟥 Image 1: Large Image with Bounding Box === #
        result_img = large_img_color.copy()  # Make a copy to draw the bounding box
        cv2.polylines(result_img, [mapped_corners], isClosed=True, color=(0, 0, 255), thickness=3)
        cv2.imwrite(OUTPUT_IMAGE_PATH, result_img)

        # === 🟩 Image 2: Side-by-Side Matching Visualization === #
        # Resize template to match the height of large image
        h_large, w_large, _ = large_img_color.shape
        scale_factor = h_large / h_template
        resized_template = cv2.resize(template_img, (int(w_template * scale_factor), h_large))

        # Convert resized template to color
        resized_template_color = cv2.cvtColor(resized_template, cv2.COLOR_GRAY2BGR)

        # Stack images horizontally
        match_img = np.hstack((resized_template_color, large_img_color))

        # Adjust keypoints positions due to resizing
        for pt1, pt2 in zip(mkpts1, mkpts2):
            x1, y1 = int(pt1[0] * scale_factor), int(pt1[1] * scale_factor)  # Scale template keypoints
            x2, y2 = int(pt2[0]) + resized_template.shape[1], int(pt2[1])  # Offset x2 for side-by-side view
            cv2.line(match_img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green lines

        # Save match visualization
        cv2.imwrite(MATCH_VISUALIZATION_PATH, match_img)

        # === 🎯 Display Images with Matplotlib and Save as One === #
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Show Image 1: Bounding Box on Large Image
        axs[0].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Detected Region in Large Image")
        axs[0].axis("off")

        # Show Image 2: Keypoint Matching Lines
        axs[1].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Feature Matching Visualization")
        axs[1].axis("off")

        plt.savefig("combined_visualization.png")  # Save as a combined visualization
        plt.show()

        print(f"Bounding box saved as {OUTPUT_IMAGE_PATH}")
        print(f"Feature matching visualization saved as {MATCH_VISUALIZATION_PATH}")
        print(f"Combined visualization saved as combined_visualization.png")

    else:
        print("Not enough LoFTR keypoints found. Try using ORB/SIFT as an alternative.")

if __name__ == "__main__":
    match_template()
