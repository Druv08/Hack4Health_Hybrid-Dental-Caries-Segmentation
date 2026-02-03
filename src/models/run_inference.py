import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.attention_unet import AttentionUNet
from src.postprocessing.refine_mask import postprocess_mask as refine_mask


def load_model(checkpoint_path, device):
    model = AttentionUNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to load image.")

    image = cv2.resize(image, image_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=(0, 1))
    return torch.from_numpy(image)


def threshold_mask(mask, threshold=0.5):
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255
    return mask


def create_overlay(image, mask):
    """
    Create overlay with caries regions highlighted in red.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = [0, 0, 255]  # Red in BGR
    return overlay


def get_input_image():
    """
    Find the single image in input_image/ folder.
    """
    input_dir = "input_image"
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Input folder '{input_dir}' not found. "
            "Please create it and place one image inside."
        )
    
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    images = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ]
    
    if len(images) == 0:
        raise FileNotFoundError(
            f"No images found in '{input_dir}'. "
            "Please place one dental X-ray image inside."
        )
    
    if len(images) > 1:
        print(f"[WARNING] Multiple images found. Using: {images[0]}")
    
    return os.path.join(input_dir, images[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "results/checkpoints/best_model.pth"
    
    # Get input image from input_image/ folder
    input_image_path = get_input_image()
    print(f"[INFO] Processing image: {input_image_path}")

    os.makedirs("results/predictions", exist_ok=True)

    model = load_model(checkpoint_path, device)

    image_tensor = preprocess_image(input_image_path).to(device)
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    original_resized = cv2.resize(original_image, (256, 256))

    with torch.no_grad():
        output = model(image_tensor)

    raw_mask = threshold_mask(output)
    mask = refine_mask(raw_mask)
    overlay = create_overlay(original_resized, mask)

    cv2.imwrite("results/predictions/predicted_mask.png", mask)
    cv2.imwrite("results/predictions/overlay.png", overlay)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_resized, cmap="gray")
    plt.title("Original X-ray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/predictions/visualization.png", dpi=150)
    plt.show()

    print("[OK] Inference complete. Results saved to results/predictions/")


if __name__ == "__main__":
    main()
