import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_MODEL = AutoModel.from_pretrained("facebook/dinov2-large").to(DEVICE)
DINO_PROCESSOR = AutoProcessor.from_pretrained("facebook/dinov2-large")
RESNET_MODEL = models.resnet50(pretrained=True).to(DEVICE).eval()
SWIN_MODEL = AutoModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").to(DEVICE)
SWIN_PROCESSOR = AutoProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
MOBILENET_MODEL = models.mobilenet_v3_large(pretrained=True).to(DEVICE).eval()

# Feature extraction function
def load_and_extract_features(image, model, processor=None):
    """
    Extract features from an image using the specified model.
    """
    if model in [DINO_MODEL, SWIN_MODEL]:
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].squeeze(0)  # Extract CLS token features
    elif model in [RESNET_MODEL, MOBILENET_MODEL]:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = model.features(img_tensor) if model == MOBILENET_MODEL else model.conv1(img_tensor)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # Adaptive pooling
            features = features.view(features.size(0), -1).squeeze(0)  # Flatten feature map
    return torch.nn.functional.normalize(features, p=2, dim=-1)

# Compute similarity per extractor
def compute_similarity_per_extractor(template_features, window_features):
    """
    Compute cosine similarity per extractor.
    """
    similarities = [torch.cosine_similarity(template_features[i], window_features[i], dim=0).item() for i in range(len(template_features))]
    return similarities

# Sliding window search
def sliding_window_search(template_path, lsr_path, stride=5, weights=[0.2, 0.2, 0.3, 0.3]):
    """
    Perform template matching using a sliding window approach with multiple feature extractors.
    """
    # Load images
    template_image = Image.open(template_path).convert("RGB")
    lsr_image = cv2.imread(lsr_path)
    
    lsr_pil = Image.fromarray(cv2.cvtColor(lsr_image, cv2.COLOR_BGR2RGB))
    
    # Extract template features
    template_features = [
        load_and_extract_features(template_image, DINO_MODEL, DINO_PROCESSOR),
        load_and_extract_features(template_image, SWIN_MODEL, SWIN_PROCESSOR),
        load_and_extract_features(template_image, RESNET_MODEL),
        load_and_extract_features(template_image, MOBILENET_MODEL)
    ]
    
    # Get template size
    template_width, template_height = template_image.size
    lsr_height, lsr_width, _ = lsr_image.shape
    
    max_similarity = -1
    best_coords = None
    
    # Heatmap storage
    similarity_per_extractor = [[] for _ in range(len(template_features))]
    candidate_boxes = [[] for _ in range(len(template_features))]
    x_positions = []
    window_count = 0  # Track the window index
    
    # Sliding window search
    for yi, y in enumerate(range(0, lsr_height - template_height + 1, stride)):
        for xi, x in enumerate(range(0, lsr_width - template_width + 1, stride)):
            window = lsr_image[y:y + template_height, x:x + template_width]
            window_pil = Image.fromarray(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
            
            # Extract window features
            window_features = [
                load_and_extract_features(window_pil, DINO_MODEL, DINO_PROCESSOR),
                load_and_extract_features(window_pil, SWIN_MODEL, SWIN_PROCESSOR),
                load_and_extract_features(window_pil, RESNET_MODEL),
                load_and_extract_features(window_pil, MOBILENET_MODEL)
            ]
            
            # Compute similarity
            similarities = compute_similarity_per_extractor(template_features, window_features)
            x_positions.append(window_count)
            window_count += 1
            
            for i in range(len(similarities)):
                similarity_per_extractor[i].append((similarities[i], x, y))
    
    # Select top-3 candidates for each extractor
    for i in range(len(similarity_per_extractor)):
        similarity_per_extractor[i].sort(reverse=True, key=lambda x: x[0])
        candidate_boxes[i] = similarity_per_extractor[i][:3]
    
    # Plot top-3 matches per extractor
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    extractors = ["DINOv2", "Swin Transformer", "ResNet", "MobileNet"]
    for i, ax in enumerate(axes.flat):
        img_copy = lsr_image.copy()
        for _, x, y in candidate_boxes[i]:
            cv2.rectangle(img_copy, (x, y), (x + template_width, y + template_height), (0, 255, 0), 2)
        ax.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Top-3 Matches - {extractors[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("top3_matches_per_extractor.png")
    plt.show()
    
    return best_coords, max_similarity

if __name__ == "__main__":
    template_path = "./template.bmp"
    lsr_path = "./target.bmp"
    best_coords, similarity_score = sliding_window_search(template_path, lsr_path, stride=10, weights=[0.2, 0.2, 0.3, 0.3])
