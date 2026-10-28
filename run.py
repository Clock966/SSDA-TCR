'''
Pre-training
supervised training (source domain) + contrastive self supervised training (target domain)
ECAPA-TDNN
'''

from omegaconf import OmegaConf
import os
from model import ECAPA_TDNN
from wav_dset_1 import wav_dset_1
from wav_dset_2 import wav_dset_2
import torch
from trainer import Trainer
from loss.triplet import LossFunction
#from loss.angleproto import LossFunction
#from loss.proto import LossFunction
#from loss.ge2e import LossFunction
from loss.ap_loss import lossFunction
from loss.emp_ssl_loss import EMPSSLLoss
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load config file
cfg = OmegaConf.load('./config.yaml')

# create folders
checkpoint_save_path = os.path.join(cfg.save_path, 'checkpoint')
checkpoint_save_path_t = os.path.join(cfg.save_path, 'checkpoint_t')
logger_save_path = os.path.join(cfg.save_path, 'logger')
os.makedirs(checkpoint_save_path, exist_ok=True)
os.makedirs(checkpoint_save_path_t, exist_ok=True)
os.makedirs(logger_save_path, exist_ok=True)

# initialise
model1= ECAPA_TDNN(C=1024, n_class=cfg.n_class, m=cfg.m, s=cfg.s)
model2= ECAPA_TDNN(C=1024, n_class=cfg.n_class, m=cfg.m, s=cfg.s)
path = './campplus_cn_en_common.pt'
# #
weight = torch.load(path)
model_dict = model1.state_dict()
state_dict = {}

for key,value in model_dict.items():
    for k,v in weight.items():
        if k in key:
            state_dict[key] = v
model_dict.update(state_dict)
model1.load_state_dict(model_dict)
model2.load_state_dict(model_dict)
# #
# for name,param in model1.named_parameters():
#     if 'dense' in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

for name,param in model2.named_parameters():
    param.requires_grad = False

# model_dict = torch.load('/home/zjh/download/sv-constract/1-Pre_training/super_clean/checkpoint/model_7.pt')
# model.load_state_dict(model_dict)
Loss = LossFunction(init_w=cfg.init_w, init_b=cfg.init_b)
#Loss = lossFunction(init_w=cfg.init_w, init_b=cfg.init_b)
#Loss = EMPSSLLoss()
# train_dataset_1 = wav_dset_1(cfg.train_list_vox2, cfg)
# train_loader_1 = torch.utils.data.DataLoader(train_dataset_1,
#                                            batch_size=cfg.batch_size_1,
#                                            shuffle=True,
#                                            num_workers=cfg.n_thread,
#                                            drop_last=True,
#                                            )
train_dataset_2 = wav_dset_2(cfg)
train_loader_2 = torch.utils.data.DataLoader(train_dataset_2,
                                           batch_size=cfg.batch_size_2,
                                           shuffle=True,
                                           num_workers=cfg.n_thread,
                                           drop_last=True,
                                           )
parameters_m = model1.parameters()
parameters_l = Loss.parameters()
parameters = list(parameters_m) + list(parameters_l)
optim = torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.step_size, gamma=cfg.lr_decay)

trainer = Trainer(cfg,
                  model1=model1,
                  model2=model2,
                  Loss=Loss,
                  #train_loader_1=train_loader_1,
                  train_loader_2=train_loader_2,
                  optim=optim,
                  scheduler=scheduler,
                  )

# start training
trainer.train()