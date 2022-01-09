import os
import sys
import json
import torch
import torch.nn as nn
import argparse

from dataset_vctk_cpy import SpeakerMetaDataset 
from model import transformerencoder_maml_spkemb_mask_02
from ossem_vctk_emb import maml

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default=os.getcwd(), type=str)
    parser.add_argument('--exp_name', default='logs', type=str)
    parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--meta', default='maml', type=str)
    args = parser.parse_args()

    ##############
    ## TRAINING ##
    ##############
    # Set parameters.
    meta_mode = args.meta
    n, k = 1, 1
    query = 20 # 20
    num_inner_loop = 5
    num_inner_loop_test = 5
    inner_lr = 3e-2 #scratch
    outer_lr = 1e-4 
    #inner_lr = 3e-4 #pre
    #outer_lr = 1e-6
    num_batch = 1  # 4
    max_iter = int(100000) #132000
    use_cuda = True
    

    ckpt_path = os.path.join(args.exp_dir, args.exp_name, 'ckpt')
    # Define model. You can use any neural network-based model.
    #model = transformerencoder_03()
    #model = transformerencoder_maml_spkemb()
    model = transformerencoder_maml_spkemb_mask_02()
    #model = LSTM()
    # Define loss function.
    loss_f = torch.nn.L1Loss()
    # Define MAML.
    maml_model = maml(meta_mode, n, k, model, loss_f, num_inner_loop, inner_lr, outer_lr, use_cuda)
    
    if args.mode == "train":
        
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
            os.makedirs(ckpt_path.replace('ckpt', 'logs'))

            with open(os.path.join(ckpt_path, 'hparams.json'), 'w') as f:
                json.dump(vars(args), f)
        
        else:
            print(f'Experiment {args.exp_name} already exists.')
            sys.exit()
        
        # Load training dataset.
        tr_dataset = SpeakerMetaDataset(batchsz=max_iter // 100, resize=63, n_way = n, k_query = query)
        # Fit the model according to the given dataset.
        maml_model.fit(tr_dataset, num_batch, args, ckpt_path)
    
    elif args.mode == "test":
        ##########
        ## TEST ##
        ##########
        # Load test dataset.
        maml_model.eval()
        # Predict and calculate accuracy.
        score = maml_model.prediction(args, ckpt_path, num_inner_loop_test)
        print("ENH:  ", score)
