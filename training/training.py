from models.vgg import *
from models.utils import *
from models.blocks import *
from models.discriminator import *
from models.generator import *
from models.embedder import *

import os
import json 
import numpy as np
import time
import datetime
import itertools
import torch
import torch.nn as nn
from torchvision.utils import save_image

from training.logger import Logger

class Trainer():

    def __init__(self, device, train, directory, dataset, path_to_data, batch_size, size, path_to_finetuning_data, meta_learned_path, meta_learned_model_path, num_epoch, resume_epoch, restored_model_path, lr_G, lr_D, weight_decay, beta1, beta2, milestones, scheduler_gamma, g_adv_weight, g_vgg19_weight, g_vggface_weight, g_match_weight, g_fm_weight, d_adv_weight, print_freq, sample_freq, model_save_freq, test_path, test_model_path):
        
        self.device =device
        self.train_bool = train

        ##############
        # Directory Setting
        ###############
        self.directory = directory
        log_dir = os.path.join(directory, "logs")
        sample_dir = os.path.join(directory, "samples")
        result_dir = os.path.join(directory, "results")
        model_save_dir = os.path.join(directory, "models")
        Wi_save_dir = os.path.join(directory, "Wi")
        if not os.path.exists(os.path.join(directory, "logs")):
            os.makedirs(log_dir)
        self.log_dir = log_dir

        if not os.path.exists(os.path.join(directory, "samples")):
            os.makedirs(sample_dir)
        self.sample_dir = sample_dir

        if not os.path.exists(os.path.join(directory, "results")):
            os.makedirs(result_dir)
        self.result_dir = result_dir

        if not os.path.exists(os.path.join(directory, "models")):
            os.makedirs(model_save_dir)
        self.model_save_dir = model_save_dir

        if not os.path.exists(os.path.join(directory, "Wi")):
            os.makedirs(Wi_save_dir)
        self.Wi_save_dir = Wi_save_dir
    
        self.dataset = dataset
        self.path_to_data = path_to_data

        self.batch_size = batch_size
        self.curr_size = 224
        self.size = size

        self.path_to_finetuning_data = path_to_finetuning_data
        self.meta_learned_path = meta_learned_path
        self.meta_learned_model_path = meta_learned_model_path

        self.num_epoch = num_epoch
        self.resume_epoch = resume_epoch
        self.restored_model_path = restored_model_path

        self.lr_G = lr_G
        self.lr_D = lr_D
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.milestones = milestones
        self.scheduler_gamma = scheduler_gamma

        self.g_adv_weight = g_adv_weight
        self.g_vgg19_weight = g_vgg19_weight
        self.g_vggface_weight= g_vggface_weight
        self.g_match_weight =g_match_weight
        self.g_fm_weight =g_fm_weight

        self.d_adv_weight = d_adv_weight

        self.print_freq = print_freq
        self.sample_freq = sample_freq
        self.model_save_freq = model_save_freq
        
        self.test_path = test_path
        self.test_model_path =test_model_path
        
        self.ReLU = nn.ReLU()
        self.L1_Norm = nn.L1Loss()


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def build_model(self):
        self.vgg19 = VGG_19()
        self.vgg19.eval()
        self.vgg19.to(self.device)

        self.vggface = VGG_FACE()
        self.vggface.load_weights()
        self.vggface.eval()
        self.vggface.to(self.device)

        return 

    def load_model(self, model_path) :
        return

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_dir)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    def Loss_D(self, real_score, fake_score):
        loss_real = torch.mean(self.ReLU(1.0 - real_score))
        loss_fake = torch.mean(self.ReLU(1.0 + fake_score))
        # loss_d = loss_real + loss_fake
        return loss_real, loss_fake

    def Loss_CNT(self, real_X, fake_X):
        with torch.no_grad():
            vgg19_real = self.vgg19(real_X)
        with torch.enable_grad():
            vgg19_fake = self.vgg19(fake_X)
        with torch.no_grad():
            vggface_real = self.vggface(real_X)
        with torch.enable_grad():
            vggface_fake = self.vggface(fake_X)

        loss_vgg19 = 0
        for real_feat_vgg19, fake_feat_vgg19 in zip(vgg19_real, vgg19_fake):
            loss_vgg19 += self.L1_Norm(real_feat_vgg19, fake_feat_vgg19)
        
        loss_vggface = 0
        for real_feat_vggface, fake_feat_vggface in zip(vggface_real, vggface_fake):
            loss_vggface += self.L1_Norm(real_feat_vggface, fake_feat_vggface)
        
        loss_cnt = self.g_vgg19_weight * loss_vgg19 + self.g_vggface_weight * loss_vggface
        return loss_cnt
    
    def Loss_FM(self, real_disc_feats, fake_disc_feats):
        loss_fm = 0
        for real_disc_feat, fake_disc_feat in zip(real_disc_feats, fake_disc_feats):
            loss_fm += self.L1_Norm(real_disc_feat, fake_disc_feat)
        return self.g_fm_weight * loss_fm
    
    def Loss_G_Adv(self, fake_score):
        return (-1) * torch.mean(fake_score)
    
    def Loss_MCH(self, e_vectors, W):
        # W: 512 , B
        # e_vectors: B, K, 512, 1
        B, K, _ , _ = e_vectors.size ()
        W = W.unsqueeze(-1)         # 512, B, 1
        W = W.expand(512, B, K)     # 512, B, K
        W = W.permute(1, 2, 0)      # B, K, 512
        W = W.view(-1, 512)         # B*K, 512

        e_vectors = e_vectors.squeeze(-1)       # B, K, 512
        e_vectors = e_vectors.view(-1, 512)     # B*K, 512
        loss_mch = self.L1_Norm(W, e_vectors)
        return self.g_match_weight * loss_mch
        
    def train(self):
        return

    def finetune(self):
        return

    def test(self):
        return