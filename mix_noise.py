import numpy as np
import math
import os
import random
import soundfile as sf
import librosa
from tqdm import tqdm

eps = np.finfo(np.float32).eps


def rms(data):
    """
    calc rms of wav
    """
    energy = data ** 2
    max_e = np.max(energy)
    low_thres = max_e * (10 ** (-50 / 10))  # to filter lower than 50dB
    rms = np.mean(energy[energy >= low_thres])
    # rms = np.mean(energy)
    return rms

def snr_mix(clean, noise, snr):
    '''
    mix clean and noise according to snr
    '''
    clean_rms = rms(clean)
    clean_rms = np.maximum(clean_rms, eps)
    noise_rms = rms(noise)
    noise_rms = np.maximum(noise_rms, eps)
    k = math.sqrt(clean_rms / (10 ** (snr / 10) * noise_rms))
    new_noise = noise * k
    return new_noise

def mix_noise(clean, noise, snr, channels=1):
    '''
    split/pad the noise data and then mix them according to snr
    '''
    clean_length = clean.shape[0]
    noise_length = noise.shape[0]
    # st = 0  # choose the first point  # 从开头混合
    # padding the noise
    if clean_length > noise_length:   # 若干净语音更长，则循环填充noise至与clean等长
        noise = np.array((clean_length // noise_length) * list(noise) + list(noise[:(clean_length % noise_length)]))
    # split the noise
    elif clean_length < noise_length:   # 若噪声更长，则随机取部分噪声用于混合
        st = np.random.randint(noise_length + 1 - clean_length)
        noise = noise[st:st + clean_length]

    snr_noise = snr_mix(clean, noise, snr)
    return snr_noise

def get_one_spk_noise(clean, noise, snr, scale=0.5, channels=1):
    """
    mix clean and noise according to the snr and scale
    args:
        clean: numpy.array, L x C  L is always segment_length
        noise: numpy.array, L' x C
        snr: float
        scale: float
    """
    gen_noise = mix_noise(clean, noise, snr, channels)
    noisy = clean + gen_noise

    max_amp = np.max(np.abs(noisy))
    max_amp = np.maximum(max_amp, eps)
    noisy_scale = 1. / max_amp * scale
    clean = clean * noisy_scale
    noisy = noisy * noisy_scale
    return noisy, clean, noisy_scale


def get_all_files(folder):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


folder_path = "/home/zjh/download/Cluster-GuidedUDA-main/noisex-92-master/noise92"
noise_list = get_all_files(folder_path)





with open('/home/zjh/download/aishell/test.scp','r')as f:
    lines = f.readlines()

for i in tqdm(range(len(lines))):

    wav = lines[i].rstrip().split()[1]
    dirname = os.path.dirname(wav)
    choice = random.randint(0, 9)
    noise1 = noise_list[choice]

    new_dirname = dirname.replace('test','noise_test')
    os.makedirs(new_dirname,exist_ok=True)
    wav_name = wav.split('/')[-1]
    new_name = new_dirname+'/'+wav_name

    feat, fs0 = sf.read(wav)
    noisefeat1,fs1 = sf.read(noise1)
    if fs1 != fs0:
        noisefeat1 = librosa.resample(y=noisefeat1, orig_sr=fs1, target_sr=fs0)
    snrnoise1 = get_one_spk_noise(feat,noisefeat1,snr=15)
    sf.write(new_name,snrnoise1[0],16000)



