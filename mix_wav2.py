import numpy as np
import math
import soundfile as sf
import scipy.signal as sps
import librosa
import os
import warnings
import sys
import random
import argparse
import multiprocessing as mp
import traceback
eps = np.finfo(np.float32).eps
"混合语音,需要干净语音和噪声的列表,snr可指定也可随机,生成的混合语音长度与干净语音保持一致,混合之前要先生成对应的空文件夹，不然会报错"

def audioread(path, fs=16000):
    '''
    args
        path: wav path
        fs: sample rate
    return
        wave_data: L x C or L
    '''
    wave_data, sr = sf.read(path)
    wave_data = wave_data[:, 0]     # 单通道
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    return wave_data


def get_firstchannel_read(path, fs=16000):
    '''
    args
        path: wav path
        fs: sample rate
    return
        wave_data: L
    '''
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:, 0]
    return wave_data


def clip_data(data, start, segment_length):
    '''
    according the start point and segment_length to split the data
    args:
        data: numpy.array
        start: -2, -1, [0,...., L - 1]
        segment_length: int
    return:
        tgt: numpy.array
    '''
    tgt = np.zeros(segment_length)
    data_len = data.shape[0]
    if start == -2:
        """
        this means segment_length // 4 < data_len < segment_length // 2
        padding to A_A_A
        """
        if data_len < segment_length // 3:
            data = np.pad(data, [0, segment_length // 3 - data_len])
            tgt[:segment_length // 3] += data
            st = segment_length // 3
            tgt[st:st + data.shape[0]] += data
            st = segment_length // 3 * 2
            tgt[st:st + data.shape[0]] += data

        else:
            """
            padding to A_A
            """
            # st = (segment_length//2 - data_len) % 101
            # tgt[st:st+data_len] += data
            # st = segment_length//2 + (segment_length - data_len) % 173
            # tgt[st:st+data_len] += data
            data = np.pad(data, [0, segment_length // 2 - data_len])
            tgt[:segment_length // 2] += data
            st = segment_length // 2
            tgt[st:st + data.shape[0]] += data

    elif start == -1:
        '''
        this means segment_length < data_len*2
        padding to A_A
        '''
        if data_len % 4 == 0:
            tgt[:data_len] += data
            tgt[data_len:] += data[:segment_length - data_len]
        elif data_len % 4 == 1:
            tgt[:data_len] += data
        elif data_len % 4 == 2:
            tgt[-data_len:] += data
        elif data_len % 4 == 3:
            tgt[(segment_length - data_len) // 2:(segment_length - data_len) // 2 + data_len] += data

    else:
        if data_len - start < segment_length:
            start = data_len - segment_length
        tgt += data[start:start + segment_length]

    return tgt


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


def add_channels(cln_wav, channels=1):
    """
    原本为添加混响的函数
    add reverberation
    args:
        cln_wav: L
        rir_wav: L x C
        rir_wav is always [Lr, C]
        predelay is ms
    return:
        wav_tgt: L x C
    """
    wav_tgt = np.zeros([channels, cln_wav.shape[0]])
    for i in range(channels):
        wav_tgt[i] = cln_wav
    # L x C
    wav_tgt = np.transpose(wav_tgt)
    wav_tgt = wav_tgt[:cln_wav.shape[0]]
    return wav_tgt


def get_one_spk_noise(clean, noise, snr, scale, channels):
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


def generate_data(clean_path, noise_path, snr, scale, channels=1):
    clean = get_firstchannel_read(clean_path)
    noise = get_firstchannel_read(noise_path)
    clean = add_channels(clean, channels)
    noise = add_channels(noise, channels)
    inputs, labels, noisy_scale = get_one_spk_noise(clean, noise, snr, scale, channels)
    return inputs, labels


def preprocess_func(line, segment_length, result):
    try:
        path = line.strip()
        data = get_firstchannel_read(path)
        length = data.shape[0]

        if length < segment_length:
            if length * 2 < segment_length and length * 4 > segment_length:
                result.append('{} -2\n'.format(path))
            elif length * 2 > segment_length:
                result.append('{} -1\n'.format(path))
        else:
            sample_index = 0
            while sample_index + segment_length <= length:
                result.append('{} {}\n'.format(path, sample_index))
                sample_index += segment_length
            if sample_index < length:
                result.append('{} {}\n'.format(path, length - segment_length))

    except:
        traceback.print_exc()


def get_mix_config(clean_path, noise_path, config_path, snr_range=20.0, scale_range=[0.2, 0.9]):
    '''
    generate config file
        format is:  clean_path start_time noise_path rir_path snr scale
    '''

    clean_lines = open(clean_path, 'r').readlines()
    noise_lines = open(noise_path, 'r').readlines()

    wid = open(config_path, 'w')
    noise_len = len(noise_lines)

    idx = 0
    for line in clean_lines:
        clean_path = line.strip()
        noise_path = noise_lines[idx % noise_len].strip()
        # snr = np.random.uniform(*snr_range)  # snr range is [-5, 20)    控制信噪比
        # snr = random.choice(snr_range)
        snr = snr_range
        scale = np.random.uniform(*scale_range)  # scale range is [0.2, 0.9)    控制音量
        wid.write("{} {} {} {}\n".format(clean_path, noise_path, snr, scale))
        idx = idx + 1
    wid.close()


def mix_func(line, save_dir, sample_rate, channels):
    try:
        clean_path, noise_path, snr, scale = line.split(' ')
        # L x C
        inputs, clean_ref = generate_data(clean_path, noise_path,
                                          float(snr), float(scale), channels)
        clean = os.path.basename(clean_path).replace('.wav', '')
        noise = os.path.basename(noise_path).replace('.wav', '')
        seg = '%'
        #####
        snr = str(round(float(snr), 2))
        scale = str(round(float(scale), 2))
        id = clean_path.split('/')[-3]
        place = clean_path.split('/')[-2]
        utt_id = clean + seg + noise + seg + snr + seg + scale + '.wav'
    #    save_mix_path = os.path.join(save_dir, 'mix', id, place)
    # if not os.path.isdir(save_mix_path):
    #     os.makedirs(save_mix_path)
    # save_ref_path = os.path.join(save_dir, 'ref',  id, place)
    # if not os.path.isdir(save_ref_path):
    #     os.makedirs(save_ref_path)
        sf.write(os.path.join(save_dir, 'mix', id, place, utt_id), inputs, sample_rate)
        sf.write(os.path.join(save_dir, 'ref', id, place, utt_id), clean_ref, sample_rate)
    except:
        traceback.print_exc()


def get_data(config_path, save_dir, sample_rate=16000, channels=1, num_process=12):
    '''
    according to the config file to generate data and then save in save_dir
    '''
    lines = open(config_path, 'r')

    idx = 0
    pool = mp.Pool(num_process)
    for line in lines:
        # print('idx: ', idx)
        line = line.strip()
        # multiprocessing
        pool.apply_async(
            mix_func,
            args=(line, save_dir, sample_rate, channels)
        )
        idx = idx + 1
    pool.close()
    pool.join()
    lines.close()


def main(args):
    clean_path = args.clean_wav_list
    noise_path = args.noise_wav_list
    config_path = args.mix_config_path

    save_dir = args.save_dir

    # args.generate_config = 1

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(os.path.join(save_dir, 'mix')):
        os.mkdir(os.path.join(save_dir, 'mix'))
    if not os.path.isdir(os.path.join(save_dir, 'ref')):
        os.mkdir(os.path.join(save_dir, 'ref'))
    # if not os.path.isdir(os.path.join(save_dir, 'reverb_ref')):
    #     os.mkdir(os.path.join(save_dir, 'reverb_ref'))
    # if not os.path.isdir(os.path.join(save_dir, 'noreverb_ref')):
    #     os.mkdir(os.path.join(save_dir, 'noreverb_ref'))
    if args.generate_config:
        print('LOG: preparing mix config')
        if not os.path.exists(config_path):
            get_mix_config(clean_path, noise_path, config_path)

    print('LOG: generating')
    get_data(config_path, save_dir, args.sample_rate, args.channels)


if __name__ == "__main__":
    # 无混响版本
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clean_wav_list',
        type=str,
        default='/home/zjh/download/Cluster-GuidedUDA-main/train_clean.list',
        help='the list of clean wav to read'
    )
    parser.add_argument(
        '--noise_wav_list',
        type=str,
        default='/home/mx/sv-dataset/drone_wav/AVQ-audio/test-noise.list',
        help='the list of noise wav to read'
    )
    parser.add_argument(
        '--mix_config_path',
        type=str,
        default='/home/mx/sv-dataset/voxsrc_drone/eval-AVQ/eval_snr20/eval20.config',
        help='the save path of config path to save'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/home/mx/sv-dataset/voxsrc_drone/eval-AVQ/eval_snr20',
        help='the dir to save generated_data'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='the sample rate of input and output wav'
    )
    parser.add_argument(
        '--channels',
        type=int,
        default=1,
        help='the output channels of mixed wav'
    )
    parser.add_argument(
        '--generate_config',
        type=str,
        default='True',
        help='generate mix config file or not '
    )
    args = parser.parse_args()

    if args.generate_config == 'True' \
            or args.generate_config == 'true' \
            or args.generate_config == 't' \
            or args.generate_config == 'T':
        args.generate_config = True
    else:
        args.generate_config = False
    main(args)
