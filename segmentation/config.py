from easydict import EasyDict as edict

config = edict()
config.backbone = 'unet'
config.lr = 1e-3
config.momentum = 0.9
config.train_bs = 12
config.val_bs = 12
config.num_classes = 19
config.num_epochs = 51
config.start_epoch = 0
