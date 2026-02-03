dataset_path = '/content/drive/MyDrive/Dataset/'


import os

def get_image_mask_pairs(folder):
    images, masks = [], []
    for f in sorted(os.listdir(folder)):
        if 'mask' in f.lower():
            masks.append(os.path.join(folder, f))
        else:
            images.append(os.path.join(folder, f))
    return images, masks

carries_images, carries_masks = get_image_mask_pairs(os.path.join(dataset_path, 'Carries'))
normal_images, normal_masks = get_image_mask_pairs(os.path.join(dataset_path, 'Normal'))

all_images = carries_images + normal_images
all_masks = carries_masks + normal_masks

print(f"Total images: {len(all_images)}, Total masks: {len(all_masks)}")


def match_images_masks(images, masks):
    matched_images = []
    matched_masks = []

    # Build a dictionary for masks without extension
    mask_dict = {os.path.basename(m).replace('-mask','').split('.')[0]: m for m in masks}

    for img in images:
        key = os.path.basename(img).split('.')[0]  # image name without extension
        if key in mask_dict:
            matched_images.append(img)
            matched_masks.append(mask_dict[key])

    return matched_images, matched_masks

# Combine images from both folders
all_images = carries_images + normal_images
all_masks = carries_masks + normal_masks

# Match properly
all_images, all_masks = match_images_masks(all_images, all_masks)
print(f"After matching: {len(all_images)} images, {len(all_masks)} masks")


from sklearn.model_selection import train_test_split

train_imgs, test_imgs, train_masks, test_masks = train_test_split(
    all_images, all_masks, test_size=0.2, random_state=42
)

print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")


import cv2
import numpy as np

def preprocess_image(img_path, mask_path, img_size=(256,256)):
    # Load grayscale image and mask
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize
    img = cv2.resize(img, img_size)
    mask = cv2.resize(mask, img_size)

    # CLAHE (improves local contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Median blur to reduce noise
    img = cv2.medianBlur(img, 3)

    # Normalize to [0,1]
    img = img / 255.0
    mask = mask / 255.0
    mask = np.round(mask)  # Ensure binary

    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    return img, mask


def data_generator(images, masks, batch_size=4, img_size=(256,256)):
    while True:
        for i in range(0, len(images), batch_size):
            batch_imgs, batch_masks = [], []
            for j in range(i, min(i+batch_size, len(images))):
                img, mask = preprocess_image(images[j], masks[j], img_size)
                batch_imgs.append(img)
                batch_masks.append(mask)
            yield np.array(batch_imgs), np.array(batch_masks)

train_gen = data_generator(train_imgs, train_masks)
test_gen = data_generator(test_imgs, test_masks)


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Multiply, Add, Activation
from tensorflow.keras.models import Model

def attention_block(x, g, inter_channels):
    # Attention gate
    theta_x = Conv2D(inter_channels, (1,1), padding='same')(x)
    phi_g = Conv2D(inter_channels, (1,1), padding='same')(g)
    add = Add()([theta_x, phi_g])
    act = Activation('relu')(add)
    psi = Conv2D(1, (1,1), padding='same', activation='sigmoid')(act)
    return Multiply()([x, psi])

def attention_unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    # Encoder
    c1 = Conv2D(16,3,activation='relu',padding='same')(inputs)
    c1 = Conv2D(16,3,activation='relu',padding='same')(c1)
    p1 = MaxPooling2D(2)(c1)

    c2 = Conv2D(32,3,activation='relu',padding='same')(p1)
    c2 = Conv2D(32,3,activation='relu',padding='same')(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = Conv2D(64,3,activation='relu',padding='same')(p2)
    c3 = Conv2D(64,3,activation='relu',padding='same')(c3)
    p3 = MaxPooling2D(2)(c3)

    # Bottleneck
    c4 = Conv2D(128,3,activation='relu',padding='same')(p3)
    c4 = Conv2D(128,3,activation='relu',padding='same')(c4)

    # Decoder
    u5 = UpSampling2D(2)(c4)
    att5 = attention_block(c3, u5, 64)
    u5 = Concatenate()([u5, att5])
    c5 = Conv2D(64,3,activation='relu',padding='same')(u5)
    c5 = Conv2D(64,3,activation='relu',padding='same')(c5)

    u6 = UpSampling2D(2)(c5)
    att6 = attention_block(c2, u6, 32)
    u6 = Concatenate()([u6, att6])
    c6 = Conv2D(32,3,activation='relu',padding='same')(u6)
    c6 = Conv2D(32,3,activation='relu',padding='same')(c6)

    u7 = UpSampling2D(2)(c6)
    att7 = attention_block(c1, u7, 16)
    u7 = Concatenate()([u7, att7])
    c7 = Conv2D(16,3,activation='relu',padding='same')(u7)
    c7 = Conv2D(16,3,activation='relu',padding='same')(c7)

    outputs = Conv2D(1,1,activation='sigmoid')(c7)
    return Model(inputs, outputs)

model = attention_unet()


import tensorflow as tf

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2.*intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def hybrid_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

model.compile(optimizer='adam', loss=hybrid_loss, metrics=['accuracy'])


def post_process(mask, threshold=0.5, min_size=50):
    mask = (mask > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    final_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            final_mask[labels==i] = 1
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    return final_mask

def agent_summary(mask):
    lesion_area = np.sum(mask)
    if lesion_area == 0:
        return "No carious lesions detected. Teeth appear healthy."
    else:
        return f"Detected potential carious lesion(s) covering ~{lesion_area} pixels. Recommend clinical exam. Confidence moderate."


import matplotlib.pyplot as plt

def visualize_bw(img, mask, title1="Original", title2="Mask"):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.show()


# Precompute all images and masks
X_train, Y_train = preprocess_dataset(train_imgs, train_masks)
X_test, Y_test = preprocess_dataset(test_imgs, test_masks)

# Train model (1 epoch demo)
history = model.fit(
    X_train, Y_train,
    batch_size=16,  # bigger batch → faster
    epochs=1,       # demo-friendly
    validation_data=(X_test, Y_test)
)

