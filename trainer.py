import torch
import sys
import time
import os
import tqdm
import soundfile
import numpy
import torch.nn.functional as F
from tools import *
from itertools import cycle


class Trainer:
    def __init__(
            self,
            cfg,
            model1,
            model2,
            Loss,
            #train_loader_1,
            train_loader_2,
            optim,
            scheduler,
            start_epoch=0,
            ):
        self.model1 = model1.cuda()
        self.model2 = model2.cuda()
        self.Loss = Loss
        #self.train_loader_1 = train_loader_1
        self.train_loader_2 = train_loader_2
        self.optim = optim
        self.scheduler = scheduler
        self.n_epoch = cfg.n_epoch
        self.start_epoch = start_epoch
        self.epoch = self.start_epoch
        self.loss_weight = cfg.loss_weight
        self.test_step = cfg.test_step
        self.info = {'time':None, 'loss':None, 'lr':None, 'acc':None, 'EER':50.0}
        self.logger_save_path = os.path.join(cfg.save_path, 'logger')
        self.checkpoint_save_path = os.path.join(cfg.save_path, 'checkpoint')
        self.checkpoint_save_path_t = os.path.join(cfg.save_path, 'checkpoint_t')
        self.eer_threshod = cfg.eer_threshod
        self.eval_list = cfg.eval_list
        self.model2_state = self.model2.state_dict()
        self.eers = []
        #self.eval_path = cfg.eval_path
        self.mse = torch.nn.MSELoss(reduction='mean')
        
    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch+self.n_epoch):
            self.epoch = epoch
            self._train()
            self.scheduler.step()
            self.log()
            if self.epoch % self.test_step == 0:
                self._eval()
    
    def _train(self):
        self.model1.train()
        self.model2.eval()
        t_start = time.time()
        index, loss = 0, 0
        #smax = 1000

        lr = self.optim.param_groups[0]['lr']
        for num, zip_data in enumerate(zip(self.train_loader_2), start=1):

            self.optim.zero_grad()

            data_2, source_data_2,labels_2 = zip_data[0][0], zip_data[0][1], zip_data[0][2]
            data_2 = data_2.transpose(0,1)
            source_data_2 = source_data_2.transpose(0, 1)
            labels_2 = torch.LongTensor(labels_2).cuda()
            feat = []
            fix_feat = []

            for inp in data_2:
                outp= self.model1.forward(torch.FloatTensor(inp).cuda(),labels_2,domain=2)
                feat.append(outp)

            feat = torch.stack(feat,dim=1).squeeze()
            # # #
            for inp in source_data_2:
                out1= self.model2.forward(torch.FloatTensor(inp).cuda(),labels_2,domain=2)
                fix_feat.append(out1)

            feat_fixed = torch.stack(fix_feat,dim=1).squeeze()
            # #
            #nloss_2 = 0.5* self.Loss.forward(feat,feat_fixed) + 0.5*self.mse(feat_fixed,feat)
            #nloss_2 = self.Loss.forward(feat)
            nloss_2 =  0.5*self.Loss.forward(feat) +  0.5*self.mse(feat_fixed, feat)
            nloss_2.backward()

            self.optim.step()
            index += len(labels_2)
            loss += nloss_2.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(self.epoch, lr, 100 * (num / self.train_loader_2.__len__())) + \
            " Loss: %.5f \r"        %(loss/(num)))
            sys.stderr.flush()
            sys.stdout.write("\n")
        # update training information
        self.info['time'] = time.time() - t_start
        self.info['loss'] = loss/num
        self.info['lr'] = lr
        
    def log(self):
        message = (
            f"epoch {self.epoch} "
            f"time {self.info['time']:.0f} "
            f"lr {self.info['lr']:.5f} "
            f"loss {self.info['loss']:.3f} \n"
            )
        with open(os.path.join(self.logger_save_path, 'training.log'), 'a') as f:
            f.write(message)
    
    def _eval(self):
        self.model1.eval()
        files = []
        embeddings = {}
        lines = open(self.eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[0])
            files.append(line.split()[1])
        setfiles = list(set(files))
        setfiles.sort()
        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            #audio, _  = soundfile.read(os.path.join(self.eval_path,file))
            audio, _ = soundfile.read(file)
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()
            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'constant')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis = 0).astype(numpy.float)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.model1.extract(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.model1.extract(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        
        scores, labels  = [], []
        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[0]]
            embedding_21, embedding_22 = embeddings[line.split()[1]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            label = line.split()[2]
            if label=='target':
                labels.append(1)
            else:
                labels.append(0)

        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        self.eers.append(EER)
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        
        # Save evaluation results_unlabeled
        message = (
            f"epoch {self.epoch} "
            f"EER {EER:.3f} "
            f"minDCF {minDCF:.3f} "
            f"best EER: {min(self.eers):.3f}\n"
            )
        with open(os.path.join(self.logger_save_path, 'evaluation.log'), 'a') as f:
            f.write(message)
        
        # save model checkpoint
        if EER < self.info['EER'] and EER < self.eer_threshod:
            save2path = os.path.join(self.checkpoint_save_path, f'model_{self.epoch}.pt')
            #save2path_t = os.path.join(self.checkpoint_save_path_t, f'model_T_{self.epoch}.pt')
            torch.save(self.model1.state_dict(), save2path)
            # model1_state = self.model1.state_dict()
            #
            # for k,v in model1_state.items():
            #     self.model2_state[k] = 0.1 * model1_state[k] + 0.9 * self.model2_state[k]
            #
            # torch.save(self.model2.state_dict(), save2path_t)
            # self.model2.load_state_dict(self.model2_state)
            #
            # self.info['EER'] = EER