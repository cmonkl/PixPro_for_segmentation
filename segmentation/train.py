import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from IPython.display import clear_output
import sys

sys.path.append("../input/cityscapesfiles")
from . import metrics, utils
from typing import Tuple, List, Any

# train iteration
def train_one_epoch(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data,
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    device: torch.device,
    scheduler: torch.optim,
    log_imgs: bool,
) -> Tuple[float, float, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    model.train()
    epoch_train_loss = 0.0
    epoch_miou = 0.0
    num_classes = model.num_classes
    class_miou = torch.zeros(num_classes)

    for it, (img, mask) in enumerate(tqdm(train_dataloader)):
        imgs, masks = img.to(device), mask.to(device).long()
        optimizer.zero_grad()

        outputs = model(imgs)
        _, labels = torch.max(outputs, dim=1)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        # calculate train miou
        cur_epoch_miou, cur_class_miou = metrics.get_iou2(
            labels.detach().cpu(), masks.detach().cpu(), nclasses=num_classes
        )
        epoch_miou += cur_epoch_miou
        class_miou = class_miou + cur_class_miou

        if scheduler is not None:
            scheduler.step()

        # save mask predictions from the first batch
        if it == 0 and log_imgs:
            mask_list = []
            for i in range(len(img)):
                segmentation_pred = utils.draw_segmentation_map(
                    labels[i].cpu().detach().numpy(), utils.label_color_map
                )
                segmentation_true = utils.draw_segmentation_map(
                    masks[i].cpu().detach().numpy(), utils.label_color_map
                )
                mask_list.append(
                    utils.wandb_mask(
                        img[i].cpu().permute(1, 2, 0).numpy(),
                        labels[i].cpu().detach().numpy(),
                        masks[i].cpu().detach().numpy(),
                    )
                )

            cls_acc = metrics.get_cls_acc(
                labels.detach().cpu(), masks.detach().cpu(), nclasses=num_classes
            )
        del outputs, loss

    epoch_train_loss = epoch_train_loss / len(train_dataloader)
    epoch_miou = epoch_miou / len(train_dataloader)
    class_miou = class_miou / len(train_dataloader)
    return epoch_train_loss, epoch_miou, class_miou, mask_list, cls_acc


# validation iteration
def val_one_epoch(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data,
    criterion: torch.nn.Module,
    optimizer: torch.nn.Module,
    device: torch.device,
    step: int,
    logger: Any,
    log_imgs=False,
) -> Tuple[float, float, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    epoch_val_loss = 0.0
    model.eval()
    epoch_miou = 0.0
    num_classes = model.num_classes
    class_miou = torch.zeros(num_classes)

    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(val_dataloader)):
            imgs, masks = img["img"].to(device), mask.to(device).long()

            # need in case of early break
            optimizer.zero_grad()

            outputs = model(imgs)
            labels = outputs.argmax(dim=1)

            loss = criterion(outputs, masks)
            epoch_val_loss += loss.item()

            cur_epoch_miou, cur_class_miou = metrics.get_iou2(
                labels.detach().cpu(), masks.detach().cpu(), nclasses=num_classes
            )
            epoch_miou += cur_epoch_miou
            class_miou = class_miou + cur_class_miou

            if log_imgs == True and i == 0:
                mask_list = []
                for p in range(len(imgs)):
                    segmentation_pred = utils.draw_segmentation_map(
                        labels[p].cpu().numpy(), utils.label_color_map
                    )
                    segmentation_true = utils.draw_segmentation_map(
                        masks[p].cpu().numpy(), utils.label_color_map
                    )
                    mask_list.append(
                        utils.wandb_mask(
                            imgs[p].cpu().permute(1, 2, 0).numpy(),
                            labels[p].cpu().numpy(),
                            masks[p].cpu().numpy(),
                        )
                    )

                cls_acc = get_cls_acc(
                    labels.detach().cpu(), masks.detach().cpu(), nclasses=num_classes
                )

                del outputs, loss

    epoch_val_loss = epoch_val_loss / len(val_dataloader)
    epoch_miou = epoch_miou / len(val_dataloader)
    class_miou = class_miou / len(val_dataloader)

    return epoch_val_loss, epoch_miou, class_miou, mask_list, cls_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data,
    val_dataloader: torch.utils.data,
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    start_epoch: int,
    num_epochs: int,
    logger: Any,
    device: torch.device,
    scheduler=None,
    log_imgs=False,
) -> Tuple[float, float]:
    train_loss = None
    for it in range(start_epoch, start_epoch + num_epochs):
        res = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, scheduler, log_imgs
        )
        train_loss, train_miou, train_class_miou, mask_list, train_acc = res

        if it % 10 == 0:
            if log_imgs:
                val_res = val_one_epoch(
                    model,
                    val_dataloader,
                    criterion,
                    optimizer,
                    device,
                    it,
                    logger,
                    log_imgs=True,
                )
                val_loss, val_miou, val_class_miou, v_m_list, vacc = val_res
            else:
                val_res = val_one_epoch(
                    model, val_dataloader, criterion, optimizer, device, it, logger
                )
                val_loss, val_miou, val_class_miou = val_res

            log_dict_val = {
                "val_loss": val_loss,
                "val_miou": val_miou,
                "val_class_iou": dict(
                    zip(
                        [
                            segmentation_classes[p] + "_iou"
                            for p in torch.arange(19).numpy()
                        ],
                        val_class_miou.numpy(),
                    )
                ),
                "val_pred": v_m_list,
                "val_class_acc": dict(
                    zip(
                        [
                            segmentation_classes[p] + "_acc"
                            for p in torch.arange(19).numpy()
                        ],
                        vacc.numpy(),
                    )
                ),
            }
            if logger is not None:
                logger.log(log_dict_val, step=it)

        log_dict = {
            "train_loss": train_loss,
            "train_miou": train_miou,
            "train_class_iou": dict(
                zip(
                    [
                        segmentation_classes[p] + "_iou"
                        for p in torch.arange(19).numpy()
                    ],
                    train_class_miou.numpy(),
                )
            ),
            "train_pred": mask_list,
            "train_acc": dict(
                zip(
                    [
                        segmentation_classes[p] + "_acc"
                        for p in torch.arange(19).numpy()
                    ],
                    train_acc.numpy(),
                )
            ),
        }
        if logger is not None:
            logger.log(log_dict, step=it)

        clear_output(wait=True)
        print(f"iter: {it+1}\nTrain_loss: {train_loss}")
        print(f"Validation loss: {val_loss}")
        print(f"Train_iou: {train_miou}")
        print(f"Validation iou: {val_miou}")

    return train_loss, val_loss

