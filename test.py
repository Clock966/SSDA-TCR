from model import ECAPA_TDNN
import torch
import soundfile
import numpy
import numpy as np
import torch.nn.functional as F
import tqdm
import os
from tools import *

model= ECAPA_TDNN(C=1024, n_class=40, m=0.2, s=30)
# path = '/home/ll/Documents/zjh/sv-constract_new/campplus_cn_en_common.pt'
# path = '/home/ll/Documents/zjh/sv-constractive/campplus_voxceleb.bin'
path = '/home/ll/Documents/zjh/sv-adapt/1-Pre_training/vox1_epoch_100/1_cos_MSE_0001/checkpoint/model_98.pt'
weight = torch.load(path)
model_dict = model.state_dict()
state_dict = {}
#state_dict = {k:v for k,v in weight.items() if k in model_dict.keys()}
for key,value in model_dict.items():
    for k,v in weight.items():
        if k in key:
            state_dict[key] = v
model_dict.update(state_dict)
model.load_state_dict(model_dict)


model.eval()
model.cuda()

line_list = []
files = []
embeddings = {}
lines = open('/home/ll/Documents/zjh/sv-constract_new/test.trail').read().splitlines()
# lines = open('/home/ll/Documents/zjh/sv-adapt/1-Pre_training/make_cnc_data/test.trail').read().splitlines()
for line in lines:
    # wav1 = line.split()[0]
    # wav_name1 = wav1.split('/')[-1]
    # wav2 = line.split()[1]
    # wav_name2 = wav2.split('/')[-1]
    # if wav_name1 != wav_name2:
    #     if 'noisy' in wav1:
    #         wav1 = wav1.replace('noisy','clean')
    #     if 'noisy' in wav2:
    #         wav2 = wav2.replace('noisy','clean')
    #     label = line.split()[-1]
    #     new_line = wav1 + ' ' +wav2 + ' '+ label
    #     line_list.append(new_line)
    #     files.append(wav1)
    #     files.append(wav2)
    files.append(line.split()[0])
    files.append(line.split()[1])
setfiles = list(set(files))
setfiles.sort()
for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
    audio, _ = soundfile.read(file)
    # Full utterance
    data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
    # Spliited utterance matrix
    max_audio = 300 * 160 + 240
    if audio.shape[0] <= max_audio:
        shortage = max_audio - audio.shape[0]
        audio = numpy.pad(audio, (0, shortage), 'wrap')
    feats = []
    startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
    for asf in startframe:
        feats.append(audio[int(asf):int(asf) + max_audio])
    feats = numpy.stack(feats, axis=0).astype(numpy.float)
    data_2 = torch.FloatTensor(feats).cuda()
    # Speaker embeddings
    with torch.no_grad():
        embedding_1 = model.extract(data_1)
        embedding_1 = F.normalize(embedding_1, p=2, dim=1)
        embedding_2 = model.extract(data_2)
        embedding_2 = F.normalize(embedding_2, p=2, dim=1)
    embeddings[file] = [embedding_1, embedding_2]

scores, labels = [], []

for line in lines:
    wav1 = line.split()[0]
    wav2 = line.split()[1]

    embedding_11, embedding_12 = embeddings[wav1]
    embedding_21, embedding_22 = embeddings[wav2]
    # embedding_11, embedding_12 = embeddings[line.split()[0]]
    # embedding_21, embedding_22 = embeddings[line.split()[1]]
    # Compute the scores
    score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
    score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
    score = (score_1 + score_2) / 2
    score = score.detach().cpu().numpy()
    scores.append(score)
    label = line.split()[2]
    if label == 'target':
        labels.append(1)
    else:
        labels.append(0)

# Coumpute EER and minDCF
EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
print(EER,minDCF)

