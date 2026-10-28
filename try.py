import logging.handlers
import torch
import torch.nn as nn
from sympy.abc import alpha

# with open('/home/ll/Documents/zjh/sv-constract_new/select_train.scp','r')as f:
#     lines = f.readlines()
# ids = []
# for line in lines:
#     id_wav = line.split()[0]
#     if id_wav not in ids:
#         ids.append(id_wav)
# print(len(ids))
# conv1 = nn.Conv1d(10,20,kernel_size=3).eval()
# inputs = torch.tensor([[1,1],
#                        [2,2]])
# copy = inputs.repeat(2,1)
# print(copy.shape)
# with open('/home/ll/Documents/zjh/data/cnceleb/eval/test.trail','r')as f:
#     lines = f.readlines()
# with open('cnc_test.trail', 'w')as f:
#     for line in lines:
#         write_line = line.replace('/home/zjh/download/cnceleb/CN-Celeb_wav/eval','/home/ll/Documents/zjh/data/cnceleb/eval')
#         f.write(write_line)
# import random
# a = random.randint(0,1)
# print((a))
# import os
# import glob
# wavs = glob.glob(os.path.join('/home/ll/Documents/zjh/data/cnceleb/eval','*/*.wav'))#40glob.glob(os.path.join('/home/ll/Documents/zjh/data/vox1/vox1/voxceleb1/test/wav','*/*/*.wav'))
# ids = []
# for wav in wavs:
#     wav_name = wav.split('/')[-1]
#     wav_id = wav_name.split('-')[0]
#     if wav_id not in ids:
#         ids.append(wav_id)
# print(len(ids))

# wav_list = []
# speaker_list = []
# with open('/home/ll/Documents/zjh/data/vox1/vox1/trials/vox1_H_cleaned.trial') as f:
#     lines = f.readlines()
#
# for line in lines:
#     wav1, wav2, tar = line.rstrip().split()
#     if wav1 not in wav_list:
#         wav_list.append(wav1)
#     if wav2 not in wav_list:
#         wav_list.append(wav2)
#
# for wav in wav_list:
#     speaker = wav.split('/')[0]
#     if speaker not in speaker_list:
#         speaker_list.append(speaker)
#
# print(len(wav_list),len(speaker_list))

log_dino_sv = "/home/ll/Documents/zjh/sv-adapt/1-Pre_training/vox1/1_cos_MSE_0001/logger/evaluation.log"
log_dino_tcr = "/home/ll/Documents/zjh/sv-adapt/1-Pre_training/vox1/1_cos_MSE_EMP_0001_55/logger/evaluation.log"
log_ssda_tcr = "/home/ll/Documents/zjh/sv-adapt/1-Pre_training/vox1/1_cos_MSE_EMP_0001_55_select0.1/logger/evaluation.log"

def cap_eer(path):
    with open(path,'r')as f:
        lines = f.readlines()
    eer_list = []
    for line in lines:
        eer = line.rstrip().split()[3]
        eer_list.append(float(eer))
    return eer_list

import matplotlib.pyplot as plt
import numpy as np

list_dino_sv = cap_eer(log_dino_sv)#[2.035,2.035,1.410,1.385,1.535,2.035,2.070,1.080,1.315,1.160]#cap_eer(log_dino_sv)
list_dino_tcr = cap_eer(log_dino_tcr)#[1.795,1.625,1.545,1.370,1.470,1.340,1.195,1.235,1.255,1.060]#cap_eer(log_dino_tcr)
list_ssda_tcr = cap_eer(log_ssda_tcr)#[2.045,1.505,1.260,1.150,1.140,1.180,1.105,1.115,1.155,0.940]#cap_eer(log_ssda_tcr)
list_super = [1.295]*10

x = np.arange(10)
plt.figure(figsize=(8,5))
plt.grid(True,linestyle='--',alpha=0.6)
plt.plot(x,list_dino_sv,marker='o',label='DINO-SV',color='#1f77b4')
plt.plot(x,list_dino_tcr,marker='s',label='DINO-TCR',color='#ff7f0e')
plt.plot(x,list_ssda_tcr,marker='^',label='SSDA-TCR',color='#2ca02c')
plt.plot(x,list_super,'rs--', alpha=0.5, linewidth=1, label='Supervised')

plt.title('EER Comparison of DINO-SV, DINO-TCR and SSDA-TCR Methods')
plt.xlabel('Number of Epoches')
plt.ylabel('EER')

plt.legend(loc='upper right')


plt.tight_layout()
# plt.ylim(-1,1)
plt.show()
# plt.savefig('/home/ll/Desktop/eer.png',dpi=300)



