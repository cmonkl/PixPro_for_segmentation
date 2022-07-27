import matplotlib.pyplot as plt
import cv2
import torch
import wandb
from typing import Tuple, List

# color for each enumerated class
label_color_map = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

# list of classes names
segmentation_classes = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bycicle",
]

# draw segmentation mask with colors
def draw_segmentation_map(labels: np.array, label_map: List) -> np.array:
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]

    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image


# returns a dictionary of labels
def labels_dict():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    l[255] = "background"
    return l


# util function for generating interactive image mask from components
def wandb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(
        bg_img,
        masks={
            "prediction": {"mask_data": pred_mask, "class_labels": labels_dict()},
            "ground truth": {"mask_data": true_mask, "class_labels": labels_dict()},
        },
    )


# helper function to plot images
def visualize(
    image: torch.Tensor, mask: torch.Tensor, original_image=None, original_mask=None
):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title("Original image", fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title("Original mask", fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title("Transformed image", fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title("Transformed mask", fontsize=fontsize)


# helper function to get labels from scores
def get_segment_labels(img: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    model.eval()
    img = img.unsqueeze(0)
    output = torch.argmax(model(img)["out"], dim=1).detach().cpu().numpy()
    return output


# draw segmentation mask
def image_overlay(image: torch.Tensor, segmented_image: torch.Tensor) -> np.array:
    alpha = 0.6  # how much transparency to apply
    beta = 1 - alpha  # alpha + beta should equal 1
    gamma = 0  # scalar added to each sum
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image

