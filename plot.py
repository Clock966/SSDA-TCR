import os
import glob
train = []
dev = []
test = []
with open('/home/zjh/download/WSJ0-DNS3/train-wsj0.scp','r')as f:
    lines = f.readlines()
for line in lines:
    wav_id = line.split()[0]
    if wav_id not in train:
        train.append(wav_id)
with open('/home/zjh/download/WSJ0-DNS3/dev-wsj0.scp','r')as f:
    lines = f.readlines()
for line in lines:
    wav_id = line.split()[0]
    if wav_id not in dev:
        dev.append(wav_id)

with open('/home/zjh/download/WSJ0-DNS3/test_babble.scp','r')as f:
    lines = f.readlines()
for line in lines:
    wav_id = line.split()[0]
    if wav_id not in test:
        test.append(wav_id)
#
print(len(train),len(dev),len(test))#(60,6,6)
# data = 0
# import soundfile as sf
# with open('/home/zjh/download/WSJ0-DNS3/test_babble.scp','r')as f:
#     lines = f.readlines()
# for line in lines:
#     wav = line.rsplit()[-1]
#     audio,sr = sf.read(wav)
#     if len(audio)<32240:
#         data += 1
# print(data)