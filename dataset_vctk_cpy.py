import os
import pdb
import torch
import librosa
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import collections
from PIL import Image
import csv
import random



class SpeakerMetaDataset(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root="./", mode="train", batchsz=100, n_way=5, k_shot=1, k_query=15, resize=1000, startidx=0):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        self.mode = mode
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        # number of samples per set for evaluation
        self.querysz = self.n_way * self.k_query
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
            mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = torch.load 
            csvdata = self.loadCSV(os.path.join(root, mode + '_vctk.csv'))  # csv path
            self.data = []
            self.img2label = {}
            for i, (k, v) in enumerate(csvdata.items()):
                self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
                self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
            self.cls_num = len(self.data)
            self.create_batch(self.batchsz)
        else:
            self.transform = librosa.load
            csvdata = self.loadCSV(os.path.join(root, mode + '_vctk.csv'))  # csv path
            self.data = []
            self.img2label = {}
            for i, (k, v) in enumerate(csvdata.items()):
                pdb.set_trace()
                self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
                self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
            self.cls_num = len(self.data)
            self.create_batch(self.batchsz)

    def padding(self, x):
        if x.shape[1] <= self.resize:
            x = x.transpose(0,1)
            a = torch.ones(self.resize, 257)
            return pad_sequence([a, x], batch_first=True)[-1]
        
        else:
            start = torch.randint(0, x.shape[1] - self.resize, (1, ))
            end = start + self.resize
            return x[:,start:end].transpose(0,1)
    
    def magphase(self, x, mode="log1p"):
        #pdb.set_trace()
        if mode == "log1p":
            x = x[:,:,0]+x[:,:,1]*1j
            magnitude = torch.log1p(torch.abs(x))
            phase = torch.exp(1j*torch.angle(x))
            # magnitude, phase = map(lambda x: x.requires_grad_(), [magnitude, phase])
        else:
            pass
            #TODO

        return magnitude, phase

    def stft(self, x):
        return torch.stft(
            x,
            n_fft=512,
            hop_length=256,
            win_length=512,
            center=False,
            normalized=False,
            onesided=True,
            pad_mode='reflect',
            window=torch.hamming_window(512, periodic=False)
        )

    def istft(self, x):
        return torch.istft(
            x,
            n_fft=512,
            hop_length=256,
            win_length=512,
            center=False,
            normalized=False,
            onesided=True,
            window=torch.hamming_window(512, periodic=False),
            length=(x.size(2) + 1) * 256
        )

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',') 
            #pdb.set_trace()
            #next(csvreader, None) 
            for i, row in enumerate(csvreader): 
                filename = row 
                #pdb.set_trace() 
                label = row[0].split('/p')[-1][:3] 
                if label in dictLabels.keys(): 
                    dictLabels[label].append(filename) 
                else: 
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch]
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            #print(self.cls_num, self.n_way)
            selected_cls = np.random.choice(
                self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_speech_idx = np.random.choice(
                    len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_speech_idx)
                indexDtrain = np.array(
                    selected_speech_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(
                    selected_speech_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all speech filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())
            random.shuffle(support_x)
            random.shuffle(query_x)

            # append set to current sets
            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        if self.mode == "train":

            support_x = torch.FloatTensor(self.setsz, 1, self.resize, 257)
            
            support_x_wav = torch.FloatTensor(self.setsz, 1, (self.resize+1)*256)

            support_y = torch.FloatTensor(self.setsz, 1, self.resize, 257)
            
            support_y_wav = torch.FloatTensor(self.setsz, 1, (self.resize+1)*256)

            support_x_emb = torch.FloatTensor(self.setsz, 1, 192)

            query_x = torch.FloatTensor(self.querysz, 1, self.resize, 257)
            
            query_y = torch.FloatTensor(self.querysz, 1, self.resize, 257)

            query_x_emb = torch.FloatTensor(self.querysz, 1, 192) 
            
            flatten_support_x = [item
                                 for sublist in self.support_x_batch[index] for item in sublist]

            flatten_query_x = [item
                               for sublist in self.query_x_batch[index] for item in sublist]
            
            for i, path in enumerate(flatten_support_x):
                
                #print(path[0])
                x = self.transform(path[0])
                y = self.transform(path[0].replace("noisy_", "clean_"))
                folder = path[0].split('/')[-2]
                wavename = path[0].split('/')[-1]
                emb = self.transform("./ECAPA_VCTK_speaker_emb/"+folder+"/"+wavename.replace(wavename[-6:], "002.pt")) 
                
                start = torch.randint(0, x.shape[0] - self.resize - 1, (1, ))
                end = start + self.resize
                
                ### noisy
                support_x[i] = x[start:end,:].clone().detach() 

                ### clean
                support_y[i] = y[start:end,:].clone().detach() 

                ### noisy spk emb
                support_x_emb[i] = torch.tensor(emb)

            for i, path in enumerate(flatten_query_x):
                
                x = self.transform(path[0])
                y = self.transform(path[0].replace("noisy_", "clean_"))
                folder = path[0].split('/')[-2]
                wavename = path[0].split('/')[-1]
                emb = self.transform("./ECAPA_VCTK_speaker_emb/"+folder+"/"+wavename.replace(wavename[-6:], "002.pt"))
                start = torch.randint(0, x.shape[0] - self.resize - 1, (1, ))
                end = start + self.resize

                ### noisy
                query_x[i] = x[start:end,:].clone().detach() 

                ### clean
                query_y[i] = y[start:end,:].clone().detach() 
                
                ### noisy spk emb
                query_x_emb[i] = torch.tensor(emb)

            # print(support_set_y)
            # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)
            
            return support_x, support_y, query_x, query_y, support_x_emb, query_x_emb

        if self.mode == "test":

            support_x = torch.FloatTensor(self.setsz, 1, self.resize, 257)
            
            support_y = torch.FloatTensor(self.setsz, 1, self.resize, 257)
            
            query_x = torch.FloatTensor(self.querysz, 1, self.resize, 257)
            
            query_y = torch.FloatTensor(self.querysz, 1, self.resize, 257)
            
            flatten_support_x = [item
                                 for sublist in self.support_x_batch[index] for item in sublist]

            flatten_support_y = [[item[0].replace("noisy_", "clean_")] 
                                 for sublist in self.support_x_batch[index] for item in sublist]

            flatten_query_x = [item
                               for sublist in self.query_x_batch[index] for item in sublist]
            
            flatten_query_y = [[item[0].replace("noisy_", "clean_")]
                               for sublist in self.query_x_batch[index] for item in sublist]


            length_support = []
            length_query = []
            for i, path in enumerate(flatten_support_x):
                wav, _ = self.transform(path[0], sr=16000)
                wav = torch.tensor(wav)
                spec, _ = self.magphase(self.stft(wav))
                length_support.append(int(spec.shape[1]))
                support_x[i] = torch.tensor(self.padding(spec))

            for i, path in enumerate(flatten_support_y):
                wav, _ = self.transform(path[0], sr=16000)
                wav = torch.tensor(wav)
                spec, _ = self.magphase(self.stft(wav))
                support_y[i] = torch.tensor(self.padding(spec))
            
            query_xp = []
            for i, path in enumerate(flatten_query_x):
                wav, _ = self.transform(path[0], sr=16000)
                wav = torch.tensor(wav)
                spec, phase = self.magphase(self.stft(wav))
                length_query.append(int(spec.shape[1]))
                query_x[i] = torch.tensor(self.padding(spec))
                query_xp.append(phase)
            
            query_yp = []
            for i, path in enumerate(flatten_query_y):
                wav, _ = self.transform(path[0], sr=16000)
                wav = torch.tensor(wav)
                spec, phase = self.magphase(self.stft(wav))
                query_y[i] = torch.tensor(self.padding(spec))
                query_yp.append(phase)

            return support_x, support_y, query_x, query_xp, query_y, query_yp, length_support, length_query
    
    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz
