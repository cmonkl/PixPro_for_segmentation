import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from collections import OrderedDict
from typing import Any


def train_one_epoch(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    device: torch.device,
    scheduler: torch.optim,
    it: int,
    logger: Any,
) -> float:
    model.train()
    epoch_train_loss = 0.0

    # keep track of mean num images in batch
    mean_N = 0
    # whether to use down feats
    use_downsampled_feat = type(criterion) == list
    epoch_down_loss = 0.0
    epoch_up_loss = 0.0
    # thresh for logging
    log_lim = 5

    for i, (img1, img2, coords1, coords2, img) in enumerate(tqdm(train_dataloader)):
        coords1 = coords1.to(device)
        coords2 = coords2.to(device)
        optimizer.zero_grad()

        if use_downsampled_feat:
            # use additional logging for the first iteration and the first batch
            if it == 0 and i == 0 and logger is not None:
                up_feats, feats, valid_coords, log_data = model(
                    img1, img2, coords1, coords2, log_coord=True, log_down=True
                )

                proj_coords1 = log_data["proj_coords1"]
                proj_coords2 = log_data["proj_coords2"]
                pos_coords1 = log_data["pos_coords1"]
                pos_coords2 = log_data["pos_coords2"]

                ims, pos_ims = get_log_coord_imgs(
                    img.clone(), proj_coords1, proj_coords2, pos_coords1, pos_coords2
                )

                logger.log(
                    {
                        "proj coords": [logger.Image(image) for image in ims],
                        "pos pairs coords": [logger.Image(image) for image in pos_ims],
                    },
                    step=it,
                )

                to_pil = ToPILImage()
                logger.log(
                    {
                        "img1": [
                            logger.Image(to_pil(denormalize(img)))
                            for img in img1[:log_lim]
                        ],
                        "img2": [
                            logger.Image(to_pil(denormalize(img)))
                            for img in img2[:log_lim]
                        ],
                    },
                    step=it,
                )
                logger.log(
                    {
                        "sim1": [
                            logger.Image(to_pil(img))
                            for img in log_data["sim1"][:log_lim]
                        ],
                        "sim2": [
                            logger.Image(to_pil(img))
                            for img in log_data["sim2"][:log_lim]
                        ],
                    },
                    step=it,
                )
            else:
                up_feats, feats, valid_coords, log_data = model(
                    img1, img2, coords1, coords2
                )
            N = log_data["N"]
            q1, q2, k1, k2 = up_feats
            proj_q1, proj_q2, proj_k1, proj_k2 = feats

            # log loss data for the first iteration and the first batch
            if it == 0 and i == 0 and logger is not None:
                loss_downsampled, log_data_loss = criterion[0](
                    proj_q1,
                    proj_q2,
                    proj_k1,
                    proj_k2,
                    valid_coords[0],
                    valid_coords[1],
                    log_coord=True,
                )

                ims, pos_ims = get_log_coord_imgs(
                    img.clone(),
                    log_data_loss["proj_coords1"],
                    log_data_loss["proj_coords2"],
                    log_data_loss["pos_coords1"],
                    log_data_loss["pos_coords2"],
                )
                to_pil = ToPILImage()
                logger.log(
                    {
                        "dist matrix1": [
                            logger.Image(to_pil(img))
                            for img in log_data_loss["matches1"][:log_lim]
                        ],
                        "dist matrix2": [
                            logger.Image(to_pil(img))
                            for img in log_data_loss["matches2"][:log_lim]
                        ],
                        "down proj coords": [logger.Image(image) for image in ims],
                        "down pos pairs coords": [
                            logger.Image(image) for image in pos_ims
                        ],
                    },
                    step=it,
                )
            else:
                loss_downsampled, log_data_loss = criterion[0](
                    proj_q1, proj_q2, proj_k1, proj_k2, valid_coords[0], valid_coords[1]
                )

            loss_upsampled = criterion[1](q1, q2, k1, k2)

            epoch_down_loss += loss_downsampled.item()
            epoch_up_loss += loss_upsampled.item()

            loss = loss_downsampled + loss_upsampled
        else:
            q1, q2, k1, k2, N = model(img1, img2, coords1, coords2)
            loss = criterion(q1, q2, k1, k2)

        mean_N += N

        # skip computations if there's no intersection
        if loss is None:
            continue

        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        if scheduler is not None:
            scheduler.step()

        if use_downsampled_feat:
            del loss, k1, k2, feats, valid_coords, log_data_loss, log_data
        else:
            del loss, k1, k2

    if logger is not None:
        logger.log(
            {
                "base lr": scheduler.get_lr()[0],
                "num valid images in batch": mean_N / len(train_dataloader),
                "loss up": epoch_up_loss / len(train_dataloader),
                "loss down": epoch_down_loss / len(train_dataloader),
                "q1 mean var": torch.var(q1, dim=-1).mean(),
                "q2 mean var": torch.var(q2, dim=-1).mean(),
            },
            step=it,
        )

    epoch_train_loss = epoch_train_loss / len(train_dataloader)
    return epoch_train_loss


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.nn.Module,
    start_epoch: int,
    num_epochs: int,
    logger: Any,
    device: torch.device,
    scheduler: torch.optim,
):
    train_loss = None
    mean_time = 0
    for i in range(start_epoch, start_epoch + num_epochs):
        train_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, scheduler, i, logger
        )
        if logger is not None:
            logger.log({"train_loss": train_loss}, step=i)

        clear_output(wait=True)
        print(f"iter: {i+1}\nTrain_loss: {train_loss}")

        if i == (start_epoch + num_epochs // 2):
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "lr_sched": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                },
                f"model_pixpro_{i}.pt",
            )

    return train_loss

