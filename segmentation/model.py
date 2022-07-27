import torch.nn as nn
import torch
import sys

sys.path.append("../input/pixpro-files-50/pixpro_files")
from pixpro_files.model import UNet


class FCNModel(nn.Module):
    """
    Fully Convolutional 16-s version model from PixPro paper
    """

    def __init__(self, model, feat_num, num_classes, weight_path=None):
        super().__init__()
        self.num_classes = num_classes

        self.backbone_model = model()
        if weight_path is not None:
            self.load_dict(weight_path)

        self.layer0 = nn.Sequential(*list(self.backbone_model.children())[:4])
        self.layer1 = self.backbone_model.layer1
        self.layer2 = self.backbone_model.layer2
        self.layer3 = self.backbone_model.layer3
        self.layer4 = self.backbone_model.layer4

        # or use replace_stride_with_dilation = [False, False, True]
        for i in range(len(self.layer4)):
            self.layer4[i].conv2.dilation = (2, 2)
            self.layer4[i].conv2.padding = (2, 2)
            self.layer4[i].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)

        self.classification = nn.Sequential(
            nn.Conv2d(feat_num, 256, kernel_size=3, dilation=6, padding=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=6, padding=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

        self.score_pool4 = nn.Conv2d(feat_num // 2, num_classes, kernel_size=1)
        self.upsample16 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=32, stride=16, bias=False
        )

    def load_dict(self, weight_path):
        if weight_path is not None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

            if device == torch.device("cuda"):
                checkpoint = torch.load(weight_path)
            else:
                checkpoint = torch.load(weight_path, map_location="cpu")

            if weight_path.split("/")[-1].split("_")[0] == "ckpt":
                model_dict = checkpoint["model"]
                model_keys = [
                    ".".join(key.split(".")[1:])
                    for key in model_dict.keys()
                    if key.split(".")[0] == "encoder"
                ]
                weights = {key: model_dict["encoder." + key] for key in model_keys}
            else:
                model_dict = checkpoint["model_state_dict"]
                model_keys = [
                    ".".join(key.split(".")[2:])
                    for key in model_dict.keys()
                    if key.split(".")[0] == "q_encoder" and key.split(".")[1] == "net"
                ]
                weights = {
                    key: model_dict["q_encoder.net." + key] for key in model_keys
                }

            self.backbone_model.load_state_dict(weights, strict=False)

    def forward(self, input_x):
        x = self.layer0(input_x)
        x = self.layer1(x)
        x = self.layer2(x)
        x16 = self.layer3(x)
        x32 = self.layer4(x16)
        x32 = self.classification(x32)
        x16 = self.score_pool4(x16)
        pred = self.upsample16(x32 + x16)

        cx = int((pred.shape[3] - input_x.shape[3]) / 2)
        cy = int((pred.shape[2] - input_x.shape[2]) / 2)

        y_pred = pred[:, :, cy : cy + input_x.shape[2], cx : cx + input_x.shape[3]]
        return y_pred


class SegUnet(nn.Module):
    """
    Linear evaluation model for segmentation
    """

    def __init__(self, num_classes, num_feat=64):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = UNet(bn_type="vanilla")
        self.fc = nn.Conv2d(num_feat, num_classes, kernel_size=1)

    def load_weight(self, model_dict):
        keys = [key for key in model_dict.keys() if key.split(".")[0] == "q_encoder"]
        new_keys = [".".join(key.split(".")[2:]) for key in keys]
        state_dict = {new_key: model_dict[key] for key, new_key in zip(keys, new_keys)}
        self.backbone.load_state_dict(state_dict)

    def forward(self, x):
        feats = self.backbone(x)
        scores = self.fc(feats)
        return scores

