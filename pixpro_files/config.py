import torch.optim as optim 
from easydict import EasyDict as edict
from math import sqrt

config = edict()
config.model = edict()
config.model.ppm_params = edict()

config.model.proj_layers_channels = [2048, 4096, 256] # [512, 1024, 128] 
config.model.ppm_params.gamma = 2
config.model.ppm_params.num_features = 256 #128
config.model.ppm_params.transform_layers = 1
config.model.momentum = 0.99
config.model.use_instance = False

config.pos_ratio = 0.7
config.up_pos_ratio = 0.7

config.model.ppm_params.num_features = 128
config.model.ppm_params.num_up_features = 64
config.model.ppm_params.local_size = 16

config.model.proj_layers_channels = [64, 256, config.model.ppm_params.num_up_features] 
config.model.down_proj_layers_channels = [1024, 2048, config.model.ppm_params.num_features]

config.experiment_details = edict()
config.experiment_details.optimizer = edict()
config.experiment_details.optimizer.name = "sgd"
config.experiment_details.optimizer.momentum = 0.9
config.experiment_details.optimizer.weight_decay = 1e-5
config.experiment_details.optimizer.warmup = 6
config.experiment_details.num_epochs = 10

config.experiment_details.start_epoch = 149
config.experiment_details.train_batch_size = 32
config.experiment_details.optimizer.lr = 1. * config.experiment_details.train_batch_size / 256
config.experiment_details.seed = 42
config.total_num_epochs = 200

config.student_model = 'vanilla'
config.teacher_model = 'vanilla'
