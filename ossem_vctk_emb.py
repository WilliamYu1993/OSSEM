from copy import deepcopy
from collections import OrderedDict

from tqdm import tqdm
from pesq import pesq

import os
import pdb
import csv
import numpy as np
import librosa
import scipy
import importlib
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

class maml(nn.Module):
    def __init__(self, mode, n, k, model, loss, num_inner_loop, inner_lr, outer_lr, use_cuda):
        super(maml, self).__init__()
        self.mode = mode
        self.n = n
        self.k = k
        self.model = model
        self.loss = loss
        self.shot_loss = torch.nn.CosineEmbeddingLoss()
        self.num_inner_loop = num_inner_loop
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.use_cuda = use_cuda

        self.weight_name = [name for name, _ in self.model.named_parameters()]
        self.weight_len = len(self.weight_name)
        self.initialize_parameters()

        self.meta_optim = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.outer_lr
        )
        # device
        device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
        if device == 'cuda':
            print(f'DEVICE: [{torch.cuda.current_device()}] {torch.cuda.get_device_name()}')
        else:
            print(f'DEVICE: CPU') 
        
        if use_cuda:
            model.cuda()

    def creatdir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    def loadParameters(self, model, path):
        self_state = model.state_dict();
        loaded_state = torch.load(path,map_location=torch.device('cpu'))['model'];
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("__S__.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;
            
            self_state[name].copy_(param);
        return model


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

    def padding(self, x):
        len_x = x.size(-1)
        pad_len = 256 - len_x % 256
        x = F.pad(x, (0, pad_len))
        return x

    def magphase(self, x, mode="log1p"):
        if mode == "log1p":
            x = x[:,:,0]+x[:,:,1]*1j
            magnitude = torch.log1p(torch.abs(x))
            phase = torch.exp(1j*torch.angle(x))
        else:
            pass
            #TODO

        return magnitude, phase

    def magphase_np(self, x, mode="log1p"):
        x = x.numpy()
        if mode == "log1p":
            D = x[:,:,0]+x[:,:,1]*1j
            phase = np.exp(1j * np.angle(D))
            D = np.abs(D)
            magnitude = np.log1p(D)

        return magnitude, phase

    def add_gau(self, x):
        x = x.cpu()
        shape = x.shape[1]
        outshape = x.shape[2]
        
        for idx in range(shape):
            scale = torch.rand(1)*5.0+5.0
            gau = torch.normal(0.0, x[0,idx,:].std(), (outshape,))
            x[0,idx,:] = x[0,idx,:]+gau/scale

        return x

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

    def istft_np(self, x):
        return librosa.istft(x,
            center=False,
            hop_length=256,
            win_length=512,
            window=scipy.signal.hamming,
            length=(x.shape[1]+1)*256
        )

    def add_default_weights(self, weights):
        # Due to batch normalization.
        for i, name in enumerate(self.weight_for_default_names):
            weights[name] = self.weight_for_default[i]
        return weights

    def load_weights(self):
        tmp = deepcopy(self.model.state_dict())
        weights = []
        for name, value in tmp.items():
            if name in self.weight_name:
                weights.append(value.clone())
        return weights

    def forward(self, support_x, support_y, query_x, emb, q_emb, mode='maml', num_inner_loop=None, cum=False):
        if num_inner_loop is None:
            num_inner_loop = self.num_inner_loop

        n_way = self.n 

        if mode == 'maml':

            for idx in range(num_inner_loop):

                if idx > 0:
                    self.model.load_state_dict(self.updated_state_dict)
            
                weight_for_autograd = self.load_weights()

                loss_for_local_update = 0.0

                support_x_gau = self.add_gau(support_x)
                if self.use_cuda:
                    support_x = support_x.cuda()
                    support_x_gau = support_x_gau.cuda()
                    support_y = support_y.cuda()
                    emb = emb.cuda()
                #pdb.set_trace()                 
                support_y_pred = self.model(support_x, emb/100.0, mm="spk")
                loss_for_local_update = self.loss(support_y_pred, support_y)
                grad = torch.autograd.grad(
                    loss_for_local_update,
                    self.model.parameters(),
                    create_graph=True
                )

                for w_idx in range(self.weight_len):
                    if grad[w_idx] is None:
                        self.updated_state_dict[self.weight_name[w_idx]] = weight_for_autograd[w_idx].clone()
                    else:
                        self.updated_state_dict[self.weight_name[w_idx]] = weight_for_autograd[w_idx].clone() - self.inner_lr * grad[w_idx]
            

            self.model.load_state_dict(self.updated_state_dict)
        	
            query_y_pred = []
            for qdx in range(query_x.shape[0]):
                q_x = query_x[qdx]
                if self.use_cuda:
                    q_x = q_x.unsqueeze(0).cuda()
                    qemb = (q_emb[qdx]/100.0).cuda()
                
                query_y_pred.append(self.model(q_x, qemb, mm="spk"))
                q_x = q_x.cpu()
                qemb = qemb.cpu()

            self.model.load_state_dict(self.keep_weight)

        return query_y_pred, loss_for_local_update 

    def testing_fwd(self, support_x, support_y, emb, num_inner_loop=None):

        if num_inner_loop is None:
            num_inner_loop = self.num_inner_loop

        for idx in range(num_inner_loop):
            
            if idx > 0:
                self.model.load_state_dict(self.updated_state_dict_test)

            weight_for_autograd = self.load_weights()
            loss_for_local_update = 0.0

            spt_x = support_x.transpose(0,1).unsqueeze(0)
            spt_y = support_y.transpose(0,1).unsqueeze(0)
            support_y_pred = self.model(spt_x.cuda(), (emb/100.0).cuda(), mm="spk")
            loss_for_local_update = self.loss(support_y_pred, spt_y.cuda())
            spt_y = spt_y.cpu()

            grad = torch.autograd.grad(
                loss_for_local_update,
                self.model.parameters(),
                create_graph=True
            )
            
            for w_idx in range(self.weight_len):
                #pdb.set_trace()
                if grad[w_idx] is None:
                    self.updated_state_dict_test[self.weight_name[w_idx]] = weight_for_autograd[w_idx].clone()
                else:
                    self.updated_state_dict_test[self.weight_name[w_idx]] = weight_for_autograd[w_idx].clone() - self.inner_lr * grad[w_idx]

    def store_state(self):
        self.keep_weight = deepcopy(self.model.state_dict())

    def initialize_parameters(self):
        self.store_state()
        self.weight_for_default = torch.nn.ParameterList([])
        self.weight_for_default_names = []
        for name, value in self.keep_weight.items():
            if not name in self.weight_name:
                self.weight_for_default_names.append(name)
                self.weight_for_default.append(
                    torch.nn.Parameter(value.to(dtype=torch.float)).clone()
                )
        self.free_state()

    def free_state(self):
        self.updated_state_dict = OrderedDict()
        self.updated_state_dict_test = OrderedDict()
        self.updated_state_dict = self.add_default_weights(self.updated_state_dict)
        self.updated_state_dict_test = self.add_default_weights(self.updated_state_dict_test)

    def fit(self, tr_dataset, num_batch, args, ckpt_path):
        
        best_loss = 10.0 # initialize best loss by great value
        best_val_pesq = 0.0
        writer = SummaryWriter(os.path.join(args.exp_dir, args.exp_name, 'logs'))
        """
        ### Load pretrain SE baseline as initial
        pre_ckpt_path = "PRETRAINED MODEL PARAMETER PATH"
        #pdb.set_trace()
        ckpt_pre = torch.load(pre_ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt_pre['model_state_dict'])
        self.store_state()
        #pdb.set_trace()
        self.pre_pesq_avg = self.prediction(args, ckpt_path, self.num_inner_loop) # check for pretrain load
        print("Pre_PESQ_AVG: ", self.pre_pesq_avg)
        best_val_pesq = self.pre_pesq_avg
        #pdb.set_trace()
        ### end of pretrain
        """
        
        mode = "maml"

        for epoch in range(10 * 10):
             
            if epoch >= 30 and epoch < 80:   #warmup
                self.inner_lr += (1e-2/50.0)
            elif epoch < 30:
                self.inner_lr = 0.0 
            """
            if epoch >= 30:                  #cooldown
                ilr = self.inner_lr
                self.inner_lr = ilr/1.2
            """            
            db = DataLoader(tr_dataset, num_batch, shuffle=True, num_workers=5)#, pin_memory=True)
            total_loss = 0.0 # initialize for sum of loss per epoch
            log_loss = 0.0
            datasetsize = len(tr_dataset)/num_batch
            for step, (x_spt, y_spt, x_qry, y_qry, x_spt_emb, x_qry_emb) in tqdm(enumerate(db)):

                x_spt, y_spt = x_spt.squeeze(2), y_spt.squeeze(2)
                x_qry, y_qry = x_qry.squeeze(2), y_qry.squeeze(2)
                
                loss = 0.0
                batch_loss = 0.0
                for i in range(num_batch):
                    pred_query_y, spt_loss = self(x_spt[i], y_spt[i], x_qry[i], x_spt_emb[i], x_qry_emb[i], mode)
                    batch_loss += spt_loss
                    for qidx in range(x_qry[i].shape[0]):
                        if self.use_cuda:
                            y_q = y_qry[i][qidx].cuda()
                        pred_qy = pred_query_y[qidx][0]
                        loss += self.loss(pred_qy, y_q)
                        y_q = y_q.cpu()
                ### check in batch
                #print(self.prediction(args, ckpt_path, self.num_inner_loop))
                #print("loaded in batch")
                loss /= (num_batch*x_qry[i].shape[0])
                total_loss += loss # calculate sum of loss per step
                log_loss += loss
                self.meta_optim.zero_grad()
                #pdb.set_trace()
                loss.backward()
                self.meta_optim.step()
                self.store_state()

                writer.add_scalar('Loss/batch_inner_loss', batch_loss / num_batch, epoch * datasetsize + step)

                if (step+1)%10 == 0:
                    log = epoch * datasetsize + step
                    writer.add_scalar('Loss/valid_log', log_loss / 10, log)
                    log_loss = 0.0
            curr_loss = total_loss/(step+1) # calculate avg of loss per epoch
            writer.add_scalar('Loss/valid', curr_loss, epoch)

            '''
            if curr_loss < best_loss: # decide model saving per epoch
                best_loss = curr_loss
                save_path = os.path.join(ckpt_path, 'model_best.ckpt')
                print(f'Saving checkpoint to {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.meta_optim.state_dict(),
                    'loss': total_loss / step
                }, save_path)
            '''
            if epoch == 0:
                save_path = os.path.join(ckpt_path, 'model_best.ckpt')
                print(f'Saving first checkpoint to {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.meta_optim.state_dict(),
                    'loss': total_loss / step
                }, save_path)
            
            elif epoch > 0:
                ### Validation on testset
                self.model.eval()
                val_pesq = self.prediction(args, ckpt_path, self.num_inner_loop) 
                self.model.train()
                writer.add_scalar('PESQ', val_pesq, epoch)

                if val_pesq > best_val_pesq: # decide model saving per epoch
                    best_val_pesq = val_pesq
                    save_path = os.path.join(ckpt_path, 'model_best.ckpt')
                    print(f'Saving best checkpoint to {save_path}')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.meta_optim.state_dict(),
                        'loss': total_loss / step
                        }, save_path)

        self.model.load_state_dict(self.keep_weight)        
        writer.flush()
        writer.close()

    def prediction(self, args, ckpt_path, num_inner_loop_test):
        score_total = 0
        base_total = 0
        total = 0
        mode = "maml"
        ### turn on below in test mode, turn off in train mode
        save_path = os.path.join(ckpt_path, 'model_best.ckpt')
        checkpoint = torch.load(save_path, map_location='cpu') 
        ###
        if mode == "normal":
            self.model.load_state_dict(checkpoint['model_state_dict'])

        csvdata = self.loadCSV("./test_vctk.csv") # load testing data
        
        total_pesq = 0.0
        n_data = 0
        
        for i, (k, v) in enumerate(csvdata.items()):
    
            print("Speaker: ", k)
            n_data += len(v)
            
            if mode == "maml":
                self.inner_lr = 0.0             
                ### turn on below in test mode, turn off in train mode
                self.model.load_state_dict(checkpoint['model_state_dict']) # recovering meta
                ###
                ### turn on below in train mode, turn off in test mode
                #self.model.load_state_dict(self.keep_weight)
                ###
                support_noisy_path = v[0][0].replace(v[0][0].split('p')[-1][-7:],"002.wav")
                support_clean_path = support_noisy_path.replace("noisy_", "clean_")
                support_nosiy = torchaudio.load(support_noisy_path)[0]
                support_clean = torchaudio.load(support_clean_path)[0]
                emb_name = support_noisy_path.split('/')[-1].replace(".wav", ".pt")
                folder = support_noisy_path.split('/')[-2]
                emb = torch.tensor(torch.load("./ECAPA_VCTK_speaker_emb/"+folder+"/"+emb_name))      
                emb = emb.unsqueeze(0).unsqueeze(0) 
                support_noisy_spec, _ = self.magphase_np(self.stft(support_nosiy[0]))
                support_clean_spec, _ = self.magphase_np(self.stft(support_clean[0]))
                support_noisy_spec = torch.tensor(support_noisy_spec)
                support_clean_spec = torch.tensor(support_clean_spec)
                self.testing_fwd(support_noisy_spec, support_clean_spec, emb, num_inner_loop_test)
                self.model.load_state_dict(self.updated_state_dict_test)

            for path in v:
                query_x_path = path[0]
                query_y_path = path[0].replace("noisy_", "clean_")
                S = query_x_path.split('/')
                wname = S[-1]
                query_x = torchaudio.load(query_x_path)[0]
                query_y = torchaudio.load(query_y_path)[0]
                query_x_spec, query_x_phase = self.magphase_np(self.stft(self.padding(query_x).squeeze(0)))
                query_x_spec = torch.tensor(query_x_spec)

                if mode == "maml":
                    query_pred_spec = self.model(query_x_spec.transpose(0,1).unsqueeze(0).cuda(), (emb/100.0).cuda(), mm="spk").detach().cpu()
                elif mode == "normal":
                    query_pred_spec = self.model(query_x_spec.transpose(0,1).unsqueeze(0).cuda()).detach().cpu()

                query_pred_spec = torch.expm1(query_pred_spec)
                query_pred_spec = torch.clamp(query_pred_spec, 0.0, None)
                
                query_pred = np.multiply(query_pred_spec.transpose(1,2).numpy(), [query_x_phase])[0]
                query_rec = self.istft_np(query_pred)
                self.creatdir(os.path.join("Transformer_baseline_12345_wav"))
                if query_rec.shape[0] > query_y.size(-1):
                    length = query_rec.shape[0]-query_y.size(-1)
                    query_rec=query_rec[:-length]
                elif query_rec.shape[0] < query_y.size(-1):
                    query_rec = np.pad(query_rec,(1,query_y.size(1)-query_rec.size-1),"constant",0)
                if wname == "p257_183.wav":                
                    query_rec = query_rec/max(abs(query_rec))
                torchaudio.save(os.path.join("Transformer_baseline_12345_wav", wname), torch.tensor(query_rec).unsqueeze(0), sample_rate=16000)
                score = pesq(16000, query_y.squeeze().numpy(), query_rec, 'wb')
                if wname.split('.')[0][-3:] == "002":
                    print(wname)
                    pass
                else:
                    total_pesq += score
                print(score)
            
        return total_pesq / n_data
