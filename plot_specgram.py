# import matplotlib.pyplot as plt
# import numpy as np
#
# fig, ax = plt.subplots()
#
# x = np.random.normal(2, 0.3, 12)
# y = np.random.normal(5, 0.8, 12)
# # x1 = np.random.normal(2, 0.1, 4)
# # y1 = np.random.normal(5, 0.1, 4)
# # x2 = np.random.normal(3, 0.1, 4)
# # y2 = np.random.normal(6, 0.3, 4)
# # x3 = np.random.normal(4, 0.1, 4)
# # y3 = np.random.normal(5, 0.2, 4)
# markers = ['o','o','o','o', 's', 's','s','s','^','^','^','^']
# colors = ['red','red' ,'red','red', 'green', 'green', 'green', 'green', 'purple','purple','purple','purple' ]
# # markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '8', 'd']
# # colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'brown', 'gray', 'olive', 'teal']
# # 绘制散点图，并设置散点的颜色为红色，大小为 50，透明度为 0.7
# # for i in range(4):
# #     ax.scatter(x1[i], y1[i], marker=markers[i], color=colors[i], s=50, alpha=0.7)
# # for i in range(4):
# #     ax.scatter(x2[i], y2[i], marker=markers[i+4], color=colors[i+4], s=50, alpha=0.7)
# # for i in range(4):
# #     ax.scatter(x3[i], y3[i], marker=markers[i+8], color=colors[i+8], s=50, alpha=0.7)
# for i in range(12):
#     ax.scatter(x[i], y[i], marker=markers[i], color=colors[i], s=50, alpha=0.7)
# # ax.set_xlim(1.5, 4.4)
# # # 设置 y 轴的范围
# # ax.set_ylim(2.5, 6.5)
# ax.set_xlim(1, 3)
# # 设置 y 轴的范围
# ax.set_ylim(3, 7)
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.axis('off')
# plt.show()
import numpy
import random
import soundfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import glob,os,torch
from scipy import signal

noisetypes = ['noise','speech','music']
noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
noiselist = {}
augment_files   = glob.glob(os.path.join('/home/ll/Documents/zjh/data/musan','*/*/*.wav'))
for file in augment_files:
    if file.split('/')[-3] not in noiselist:
        noiselist[file.split('/')[-3]] = []
    noiselist[file.split('/')[-3]].append(file)
rir_files  = glob.glob(os.path.join('/home/ll/Documents/zjh/data/RIRS_NOISES/simulated_rirs','*/*/*.wav'))

def wav_split( path):
    audio, sr = soundfile.read(path)
    audio = numpy.pad(audio, (2, 2), mode='constant', constant_values=0)
    max_audio = 200 * 160 + 240
    audiosize = audio.shape[0]
    if audiosize <= max_audio * 2:
        pad_len = max_audio * 2 - audiosize
        audio = numpy.pad(audio, (pad_len, pad_len), mode='constant', constant_values=0)
        audiosize = audio.shape[0]
    randsize = audiosize - (max_audio * 2)

    assert randsize >= 1
    startframe = random.sample(range(0, randsize), 2)
    startframe.sort()
    startframe[1] += max_audio
    startframe = numpy.array(startframe)
    # for more permutation
    numpy.random.shuffle(startframe)
    feats = []
    for asf in startframe:
        feats.append(audio[int(asf):int(asf) + max_audio])
    segments = numpy.stack(feats, axis=0)
    return segments

def aug(noisesnr):
    with torch.no_grad():
        aug_segments = torch.FloatTensor([])
        aug_segments2 = torch.FloatTensor([])
        for i in range(0, segments.shape[0]):
            audio = segments[i, :]
            audio = numpy.stack([audio], axis=0)
            audio2 = audio
            augtype = random.randint(0, 5)
            augtype2 = random.randint(0, 5)
            if augtype == 0:  # Original
                audio = audio
            elif augtype == 1:  # Reverberation
                audio = add_rev(audio)
            elif augtype == 2:  # Babble
                audio = add_noise(audio, 'speech',numnoise,noiselist,noisesnr)
            elif augtype == 3:  # Music
                audio = add_noise(audio, 'music',numnoise,noiselist,noisesnr)
            elif augtype == 4:  # Noise
                audio = add_noise(audio, 'noise',numnoise,noiselist,noisesnr)
            elif augtype == 5:  # Television noise
                audio = add_noise(audio, 'speech',numnoise,noiselist,noisesnr)
                audio = add_noise(audio, 'music',numnoise,noiselist,noisesnr)
            if augtype2 == 0:  # Original
                audio2 = audio2
            elif augtype2 == 1:  # Reverberation
                audio2 = add_rev(audio2)
            elif augtype2 == 2:  # Babble
                audio2 = add_noise(audio2, 'speech',numnoise,noiselist,noisesnr)
            elif augtype2 == 3:  # Music
                audio2 = add_noise(audio2, 'music',numnoise,noiselist,noisesnr)
            elif augtype2 == 4:  # Noise
                audio2 = add_noise(audio2, 'noise',numnoise,noiselist,noisesnr)
            elif augtype2 == 5:  # Television noise
                audio2 = add_noise(audio2, 'speech',numnoise,noiselist,noisesnr)
                audio2 = add_noise(audio2, 'music',numnoise,noiselist,noisesnr)
            audio = torch.FloatTensor(audio[0])
            audio = audio.unsqueeze(0)
            aug_segments = torch.cat((aug_segments, audio), 0)

            audio2 = torch.FloatTensor(audio2[0])
            audio2 = audio2.unsqueeze(0)
            aug_segments2 = torch.cat((aug_segments2, audio2), 0)
            print(f"augtype: {augtype}, augtype2: {augtype2}")

            # source_segments = torch.from_numpy(segments).type(torch.float32)
    return aug_segments, aug_segments2

def add_rev(audio):
    rir_file    = random.choice(rir_files)
    rir, sr     = soundfile.read(rir_file)
    rir         = numpy.expand_dims(rir.astype(numpy.float),0)
    rir         = rir / numpy.sqrt(numpy.sum(rir**2))
    return signal.convolve(audio, rir, mode='full')[:,:200 * 160 + 240]

def add_noise(audio, noisecat,numnoise,noiselist,noisesnr):
    clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
    numnoise    = numnoise[noisecat]
    noiselist   = random.sample(noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
    noises = []
    for noise in noiselist:
        noiseaudio, sr = soundfile.read(noise)
        length = 200 * 160 + 240
        if noiseaudio.shape[0] <= length:
            shortage = length - noiseaudio.shape[0]
            noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
        noiseaudio = noiseaudio[start_frame:start_frame + length]
        noiseaudio = numpy.stack([noiseaudio],axis=0)
        noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4)
        noisesnrs   = random.uniform(noisesnr[noisecat][0],noisesnr[noisecat][1])
        noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnrs) / 10)) * noiseaudio)

    noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
    return noise + audio

# 加载音频文件
audio_path = '/home/ll/Documents/zjh/data/vox1/vox1/voxceleb1/test/wav/id10270/5r0dWxy17C8/00001.wav'  # 请将 'your_audio_file.wav' 替换为你的音频文件的实际路径
# y, sr = librosa.load(audio_path)
segments = wav_split(audio_path)
aug_segments, aug_segments2 = aug(noisesnr)
y = torch.cat([aug_segments,aug_segments2],dim=0)
inputs =y.numpy()
lens = len(y)
# 计算短时傅里叶变换（STFT）
for i in range(lens):
    y = inputs[i]
    stft = librosa.stft(y)
    # 将幅度谱转换为分贝（dB）刻度
    db = librosa.amplitude_to_db(abs(stft))
    # 绘制语谱图
    plt.figure(figsize=(1, 1))
    librosa.display.specshow(db, sr=16000, x_axis='time', y_axis='hz')
    # plt.colorbar()
    plt.axis('off')
    # plt.title('Spectrogram')
    plt.show()