import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import ImageFilter, Image
import numpy as np
from torch.utils.data import Dataset
import os


class RandomResizedCropCoord(transforms.RandomResizedCrop):
    """
    Crop patch and resize
    """

    def __init__(self, *args, **kwargs):
        # relative coords in the range [0..1]
        self.use_relative = kwargs.pop("relative_coord", False)
        super(RandomResizedCropCoord, self).__init__(*args, **kwargs)

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        width, height = img.size

        # normalized coords
        if self.use_relative:
            coords = torch.Tensor(
                [
                    float(i) / (height - 1),
                    float(j) / (width - 1),
                    float(i + h - 1) / (height - 1),
                    float(j + w - 1) / (width - 1),
                ]
            )
        else:
            coords = torch.Tensor([i, j, i + h - 1, j + w - 1])
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), coords


class RandomHorizontalFlipCoord(transforms.RandomHorizontalFlip):
    """
    Horizontal flip applied to both image and coordinates
    """

    def __call__(self, img, coord):
        f_img = img.copy()
        f_coord = coord.clone()

        if torch.rand(1) < self.p:
            f_img = F.hflip(img)
            f_coord = torch.Tensor([coord[0], coord[3], coord[2], coord[1]])
        return f_img, f_coord


class GaussianBlur(object):
    """
    Apply Gaussian blur
    """

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Compose(object):
    """
    Compose transformations that include torchvision transforms
    and transforms with coordinates
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if "RandomResizedCropCoord" in t.__class__.__name__:
                img, coord = t(img)
            elif "RandomHorizontalFlipCoord" in t.__class__.__name__:
                img, coord = t(img, coord)
            else:
                img = t(img)
        return img, coord


class ImgnetDataset(Dataset):
    """
    ImageNet Dataset consisting only of images

    Parameters:
        root_dir: directory path
        img_dir: directory path of images in root_dir
        img_size: size of patches (n x n)
        return_img: return original image along with crops when iterating through the dataset
        num_imgs: get only first num_images samples
    """

    def __init__(
        self, root_dir: str, img_dir: str, img_size=224, return_img=False, num_imgs=None
    ):
        self.img_dir = os.path.join(root_dir, img_dir)
        self.img_paths = [path for path in os.listdir(self.img_dir)]
        if num_imgs is not None:
            self.img_paths = self.img_paths[:num_imgs]
        self.return_img = return_img
        self.img_size = img_size

        self.transform_1 = Compose(
            [
                RandomResizedCropCoord(
                    (img_size, img_size), scale=(0.2, 1.0), relative_coord=True
                ),
                RandomHorizontalFlipCoord(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_2 = Compose(
            [
                RandomResizedCropCoord(
                    (img_size, img_size), scale=(0.2, 1.0), relative_coord=True
                ),
                RandomHorizontalFlipCoord(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.1),
                transforms.RandomSolarize(p=0.2, threshold=128),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_meta = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_meta)

        orig_img = Image.open(img_path).convert("RGB")

        img_1, coord_1 = self.transform_1(orig_img)
        img_2, coord_2 = self.transform_2(orig_img)

        if self.return_img:
            transf = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                ]
            )
            return img_1, img_2, coord_1, coord_2, transf(orig_img)
        else:
            return img_1, img_2, coord_1, coord_2

