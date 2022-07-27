from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import os
import cv2
import torch
from torch.nn.functional import pad as Pad
from tqdm import tqdm
from typing import Any, Tuple, Dict

# from 30 classes to 19
label2ind = {
    0: 255,
    1: 255,
    2: 255,
    3: 255
    4: 255,
    5: 255,
    6: 255,
    7: 0,
    8: 1,
    9: 255,
    10: 255,
    11: 2,
    12: 3,
    13: 4,
    14: 255,
    15: 255,
    16: 255,
    17: 5,
    18: 255,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 255,
    30: 255,
    31: 16,
    32: 17,
    33: 18,
    -1: 255,
}

# change labels from 30 classes to 19
def encode_labels(mask: np.array) -> np.array:
    label_mask = np.zeros_like(mask)
    for k in label2ind:
        label_mask[mask == k] = label2ind[k]
    return label_mask


class CityscapesDataset(Dataset):
    """
    A Dataset Class for Cityscapes

    Inputs:
        imgs_path: img directory path
        gt_path: labels directory path
        mode: train/val/test mode
        transform: data augmentations
    """

    def __init__(self, imgs_path: str, gt_path: str, mode: str, transform: Any):
        if mode not in ["train", "val", "test"]:
            raise ValueError

        self.mode = mode
        folders_path = os.path.join(imgs_path, mode)
        mask_folder_path = os.path.join(gt_path, mode)
        self.transform = transform
        self.img_paths = []
        self.mask_paths = []

        # weights for classes in weighted Cross-Entropy loss
        self.weights_best = torch.tensor(
            [
                0.8373,
                0.9180,
                0.8660,
                1.0345,
                1.0166,
                0.9969,
                0.9754,
                1.0489,
                0.8786,
                1.0023,
                0.9539,
                0.9843,
                1.1116,
                0.9037,
                1.0865,
                1.0955,
                1.0865,
                1.1529,
                1.0507,
            ]
        )
        self.weights_median = torch.tensor(
            [
                0.0238,
                0.1442,
                0.0384,
                1.3385,
                1.0000,
                0.7148,
                4.2218,
                1.5914,
                0.0551,
                0.7577,
                0.2183,
                0.7197,
                6.4924,
                0.1254,
                3.2801,
                3.7300,
                3.7667,
                8.8921,
                2.1195,
            ]
        )
        self.weights_inv_freq = torch.tensor(
            [
                2.7123,
                16.4339,
                4.3813,
                152.5790,
                113.9898,
                81.4769,
                481.2443,
                181.3994,
                6.2780,
                86.3695,
                24.8819,
                82.0373,
                740.0728,
                14.2968,
                373.8939,
                425.1855,
                429.3611,
                1013.6041,
                241.6003,
            ]
        )
        self.class_counts = [
            431599680,
            71244208,
            266926496,
            7674642,
            10273284,
            14362517,
            2429907,
            6449012,
            186235856,
            13560382,
            46857060,
            14269284,
            1581470,
            81906408,
            3131332,
            2753076,
            2727328,
            1155111,
            4846022,
        ]

        self.num_classes = len(self.class_counts)

        for folder in os.listdir(folders_path):
            cur_folder = os.path.join(folders_path, folder)
            cur_mask_folder = os.path.join(mask_folder_path, folder)

            for file in os.listdir(cur_folder):
                self.img_paths.append(os.path.join(cur_folder, file))
                gt_name = file.split("_leftImg8bit")[0] + "_gtFine_labelIds.png"
                gt_path = os.path.join(cur_mask_folder, gt_name)
                if os.path.isfile(gt_path):
                    self.mask_paths.append(gt_path)
                else:
                    raise Exception("data find error")

    def get_class_counts(self):
        counts = torch.zeros(self.num_classes)
        for idx in tqdm(range(len(self.mask_paths))):
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
            labels = encode_labels(mask)
            for i in range(self.num_classes):
                counts[i] += (labels == i).sum().item()
        return counts

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        orig_img = cv2.imread(self.img_paths[idx])
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        mask = encode_labels(mask)

        img = orig_img
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
            if self.mode != "val":
                h, w = self.transform[0].height, self.transform[0].width
                orig_img = cv2.resize(orig_img, (w, h))

        # for val mode return image without augmentations also
        if self.mode == "val":
            img = {"img": img, "orig": orig_img}

        return img, mask
  
