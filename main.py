import sys
    
use_colab = False
use_logging = True
eval_mode = False
load_weight = True

weights_path = '../input/pixpro-files-50/model_unet_pad_100.pt'

if use_colab:
    from google.colab import drive
    drive.mount('/content/drive')

    import os
    from google.colab import files
    import zipfile
    import os

    files.upload()
    os.environ['KAGGLE_CONFIG_DIR'] = "/content"
    
if use_colab:
    !kaggle datasets download -d monkrld/imagenet-10percent
    zip_ref = zipfile.ZipFile('imagenet-10percent.zip', 'r') #Opens the zip file in read mode
    zip_ref.extractall('/tmp') #Extracts the files into the /tmp folder
    zip_ref.close()
    !rm imagenet-10percent.zip
    
if use_colab:
    !kaggle datasets download -d monkrld/pixpro-files-50
    zip_ref = zipfile.ZipFile('pixpro-files-50.zip', 'r') #Opens the zip file in read mode
    zip_ref.extractall('/tmp/pixpro') #Extracts the files into the /tmp folder
    zip_ref.close()
    !rm pixpro-files-50.zip
    
if use_colab:
    sys.path.insert(0,'/tmp/pixpro')

    %cd /content/drive/My\ Drive/Colab\ Notebooks/pixpro
else:
    sys.path.insert(0,'../input/pixpro-files-50')
    
from config import config
from data import ImgnetDataset
from utils import *
from model import PixProLocPos
from loss import ConsistencyLoss, PosConsistencyLoss
from train import train
from scheduler import get_scheduler
from lars import LARS, add_weight_decay
from validate import pixel_retrieval, RetrievalDataset


if use_colab:
    root_dir = '/tmp'
else:
    root_dir = '../input/imagenet-10percent'
dataset = ImgnetDataset(root_dir=root_dir, 
                        img_dir='train_small', 
                        img_size=128, return_img=True,
                       num_imgs=None)
                       
config.len_dataset = len(dataset)
if use_logging:
    if not use_colab:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api = user_secrets.get_secret("wandb_api") 
        !pip install --upgrade -q wandb
    else:
        !pip install --upgrade -q wandb
        !wandb login

    import wandb
    if not use_colab:
        wandb.login(key=wandb_api)

    run = wandb.init(project='pixpro',
                    config=config,
                    group='unet pad', resume='must',
                    job_type='train')

    logger = wandb
else:
    logger = None
    
    
if load_weight:
    checkpoint = torch.load(weights_path, 
                 map_location='cpu')
    config.experiment_details.start_epoch = checkpoint['epoch']
    print(config.experiment_details.start_epoch)
    
import torch.optim as optim 
from torch.utils.data import Subset, DataLoader
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataloader = DataLoader(dataset, 
                              batch_size=config.experiment_details.train_batch_size,
                              num_workers=2, pin_memory=True, shuffle=True, drop_last=True)

model = PixProLocPos(config, use_down_feat=False).to(DEVICE)
criterion = PosConsistencyLoss(config.up_pos_ratio)
#criterion = [ConsistencyLoss(config.up_pos_ratio), PosConsistencyLoss(config.pos_ratio)]

num_epochs = config.experiment_details.num_epochs

n_iter_per_epoch = config.len_dataset / config.experiment_details.train_batch_size

params = add_weight_decay(model, config.experiment_details.optimizer.weight_decay)
optimizer = torch.optim.SGD(
    params, #model.parameters(), 
    lr=config.experiment_details.optimizer.lr,
    weight_decay=config.experiment_details.optimizer.weight_decay,
    momentum=config.experiment_details.optimizer.momentum,)
optimizer = LARS(optimizer)

scheduler = get_scheduler(optimizer, n_iter_per_epoch, 
                         config.experiment_details.optimizer.warmup, 
                        config.total_num_epochs)
                        
if load_weight:
    model.load_state_dict(checkpoint['model_state_dict'])   
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    scheduler.load_state_dict(checkpoint['lr_sched'])
    
if not eval_mode:
    if use_logging:
        logger.watch(model, log='all', log_freq=n_iter_per_epoch)

    fin_train_loss = train(
        model,
        train_dataloader,
        criterion,
        optimizer,
        config.experiment_details.start_epoch,
        config.experiment_details.num_epochs,
        logger,
        DEVICE,
        scheduler
    )

    if use_logging:
        logger.finish()
        
        
if eval_mode:
    retr_data = RetrievalDataset(root_dir=root_dir, 
                            img_dir='train_small', 
                            img_size=128)
    batch_size=64
    dataloader = DataLoader(retr_data, batch_size=batch_size, shuffle=False)
    img, sim_scores, scores = pixel_retrieval(model.q_encoder.net, 
                    retr_data, dataloader, DEVICE, batch_size, logger, 0)
else:
    EPOCH = num_epochs
    torch.save({
                'epoch': EPOCH + config.experiment_details.start_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': fin_train_loss, 
                'lr_sched':scheduler.state_dict() if scheduler is not None else None,
                }, f'model_unet_pad_{EPOCH + config.experiment_details.start_epoch}.pt')
