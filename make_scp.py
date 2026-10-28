import os
import glob

wav_s = glob.glob(os.path.join('/home/ll/Documents/zjh/3D-Speaker-main/plot_tsne_data','*/*/*.wav'))

with open('/home/ll/Documents/zjh/3D-Speaker-main/plot_tsne_data/wav_for_plot.scp','w')as f:
    for wav in wav_s:
        wav_id = wav.split('/')[-3]

        line = wav_id +' '+ wav+'\n'
        f.write(line)