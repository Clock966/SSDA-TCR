from model import ECAPA_TDNN
import torch
import soundfile
import numpy
import numpy as np
import torch.nn.functional as F
import tqdm
import os
from tools import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model= ECAPA_TDNN(C=1024, n_class=40, m=0.2, s=30)
path = '/home/ll/Documents/zjh/sv-adapt/1-Pre_training/vox1_epoch_100/1_cos_MSE_EMP_0001_select0.1/checkpoint/model_74.pt'
###加载训练好的模型
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
##可以自己制作绘图的.trail文件
lines = open('/home/ll/Documents/zjh/sv-adapt/plot_tsne_data_new/wav_for_plot.trail').read().splitlines()
for line in lines:

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
        embedding_1 = embedding_1.squeeze(0)
        embedding_1 = embedding_1.cpu().numpy()

    embeddings[file] = [embedding_1]
features = []
id_list = []
labels = []
for file in setfiles:

    wav_name = file.split('/')[-3]
    id_wav = wav_name
    if id_wav not in id_list:
        id_list.append(id_wav)
    labels.append(id_list.index(id_wav))

new_features = numpy.array([embeddings[file] for file in embeddings])
new_features = numpy.squeeze(new_features)
tsne = TSNE(n_components=2,perplexity=30,n_iter=300)
features_2d = tsne.fit_transform(new_features)
unique_labels = list(set(labels))

for label in unique_labels:
    indices = [i for i,l in enumerate(labels) if l==label]
    plt.scatter(features_2d[indices,0],features_2d[indices,1],label=label)
plt.legend(loc='lower right',fontsize=20)
# plt.title('t-SNE visualization of cluster of embeddding from KD-TTA model CAM++ on THCHS')
#plt.savefig('CL-TTA_cam++_vox.png',dpi=600)
plt.show()
