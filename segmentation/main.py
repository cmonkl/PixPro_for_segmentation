import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# flags for the enviroment
use_colab = False
use_logging = False
load_weight = False
checkpoint_path = "../input/pixpro-files-50/model_unet_pad_100.pt"


if use_colab:
    from google.colab import drive

    drive.mount("/content/drive")

    import os
    from google.colab import files
    import zipfile
    import os

    files.upload()
    os.environ["KAGGLE_CONFIG_DIR"] = "/content"

if use_colab:
    print("google")
    #%cd /content/drive/My\ Drive/Colab\ Notebooks/pixpro

    #!kaggle datasets download -d monkrld/cityscapes
    # zip_ref = zipfile.ZipFile('cityscapes.zip', 'r') #Opens the zip file in read mode
    # zip_ref.extractall('/tmp') #Extracts the files into the /tmp folder
    # zip_ref.close()
    #!rm cityscapes.zip
else:
    import sys

    sys.path.append("../input/pixpro-files-50/pixpro_files")
    sys.path.append("../input/cityscapesfiles")

from segmentation.data import CityscapesDataset
from segmentation.config import config
from segmentation.model import SegUnet
from segmentation.train import train


train_transform = A.Compose(
    [
        A.Resize(448, 992),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(448, 992),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

if use_colab:
    imgs_path = "/tmp/leftImg8bit_trainvaltest/leftImg8bit"
    gt_path = "/tmp/gtFine_trainvaltest/gtFine"
else:
    imgs_path = "../input/cityscapes/leftImg8bit_trainvaltest/leftImg8bit"
    gt_path = "../input/cityscapes/gtFine_trainvaltest/gtFine"

# datasets
train_data = CityscapesDataset(imgs_path, gt_path, "train", transform=train_transform)
val_data = CityscapesDataset(imgs_path, gt_path, "val", transform=val_transform)

# logging tool
if use_logging:
    #!pip install --upgrade -q wandb
    import wandb

    wandb.login()

    run = wandb.init(
        project="segmentation", config=config, group="supervised", job_type="train"
    )
else:
    wandb = None

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_dataloader = DataLoader(
    train_data, batch_size=config.train_bs, shuffle=True, num_workers=2, pin_memory=True
)
val_dataloader = DataLoader(
    val_data, batch_size=config.val_bs, shuffle=False, num_workers=2, pin_memory=True
)
model = SegUnet(num_classes=config.num_classes)

if load_weight:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_dict = checkpoint["model_state_dict"]
    model.load_weight(model_dict)

model = model.to(device)

# freeze all weights except linear
for param in model.backbone.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss(
    weight=train_data.weights_best.to(device), ignore_index=255
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

fin_train_loss, fin_val_loss = train(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    config.start_epoch,
    config.num_epochs,
    wandb,
    device,
    scheduler=None,
    log_imgs=use_logging,
)

torch.save(
    {
        "epoch": config.num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": fin_train_loss,
    },
    f"model_segm_{config.num_epochs}.pt",
)

