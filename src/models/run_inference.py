import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.attention_unet import AttentionUNet


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


def postprocess_mask(mask, threshold=0.5):
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255
    return mask


def create_overlay(image, mask):
    image = (image * 255).astype(np.uint8)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = [255, 0, 0]
    return overlay


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "results/checkpoints/best_model.pth"
    test_image_path = "data/processed/test/images/" + \
        os.listdir("data/processed/test/images")[0]

    os.makedirs("results/predictions", exist_ok=True)

    model = load_model(checkpoint_path, device)

    image_tensor = preprocess_image(test_image_path).to(device)
    original_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    with torch.no_grad():
        output = model(image_tensor)

    mask = postprocess_mask(output)
    overlay = create_overlay(original_image, mask)

    cv2.imwrite("results/predictions/predicted_mask.png", mask)
    cv2.imwrite("results/predictions/overlay.png", overlay)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original X-ray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Inference complete. Results saved.")


if __name__ == "__main__":
    main()
