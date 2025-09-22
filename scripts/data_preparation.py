import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Image size
IMG_SIZE = 224

# 2. Augmentation pipeline
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),

    # --- Standard Augmentations ---
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),

    # --- Robustness Augmentations ---
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Validation/Test – no augmentation, just resize + normalize
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3. Dataset paths
# Paths to your dataset
train_dir = "c:/Users/DELL/Desktop/AI_ML Research Projects/chest_xray_pneumonia/data/raw/train"
val_dir   = "c:/Users/DELL/Desktop/AI_ML Research Projects/chest_xray_pneumonia/data/raw/val"
test_dir  = "c:/Users/DELL/Desktop/AI_ML Research Projects/chest_xray_pneumonia/data/raw/test"


# 4. Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(root=val_dir, transform=test_transforms)
test_dataset  = datasets.ImageFolder(root=test_dir, transform=test_transforms)

# 5. DataLoaders (batch_size can be tuned)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

if __name__ == "__main__":
    print("✅ Data preparation script ran successfully!")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")




import matplotlib.pyplot as plt
import numpy as np
import os


os.makedirs("../outputs", exist_ok=True)


dataiter = iter(train_loader)
images, labels = next(dataiter)

# Function to show a grid of images
def imshow(imgs, labels, title="Augmentation Preview"):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))  
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        img = imgs[i] / 2 + 0.5  
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title("PNEUMONIA" if labels[i]==1 else "NORMAL")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    fig.savefig("../outputs/augmentation_preview.png") 

# Show & Save
imshow(images[:8], labels[:8])


import matplotlib.pyplot as plt
import numpy as np

 # denormalizing (reverse normalization for display)
def denormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# Get one batch of training data
images, labels = next(iter(train_loader))

# Plot first 8 images
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = denormalize(images[i])
    ax.imshow(img)
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis("off")

plt.tight_layout()
plt.show()

 