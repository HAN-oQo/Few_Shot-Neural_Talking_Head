from models.vgg import *
from models.utils import *
from models.blocks import *
from models.discriminator import *
from models.generator import *
from models.embedder import *
from tqdm import tqdm, trange
import os
import json 
import numpy as np
import time
import datetime
import itertools
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
from dataloader.dataloader import *
from dataloader.finetune_dataset import *
from dataloader.landmark_utils import *
import cv2
from training.logger import Logger

class Trainer():

    def __init__(self, device, train,finetuning, directory, dataset, path_to_data, batch_size, size, path_to_finetuning_data, meta_learned_path, meta_learned_model_path, num_vid, num_epoch, resume_epoch, restored_model_path, lr_G, lr_D, weight_decay, beta1, beta2, milestones, scheduler_gamma, g_adv_weight, g_vgg19_weight, g_vggface_weight, g_match_weight, g_fm_weight, d_adv_weight, print_freq, sample_freq, model_save_freq, test_video_path, test_model_path):
        
        self.device =device
        print(self.device)
        self.train_bool = train
        self.finetune_bool = finetuning
        if self.finetune_bool:
            if not os.path.exists(meta_learned_path):
                raise(RuntimeError("Wrong meta learned path. Need correct directory"))
            if not os.path.exists(meta_learned_model_path):
                raise(RuntimeError("Wrong meta learned model path. Need correct path to model"))
        ##############
        # Directory Setting
        ###############
        self.directory = directory
        log_dir = os.path.join(directory, "logs")
        sample_dir = os.path.join(directory, "samples")
        result_dir = os.path.join(directory, "results")
        model_save_dir = os.path.join(directory, "models")
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
        self.K = 8
        self.dataset = dataset
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.curr_size = 224
        self.size = size
        self.restored_batch_idx = 0
        self.path_to_finetuning_data = path_to_finetuning_data
        self.meta_learned_path = meta_learned_path
        self.meta_learned_model_path = meta_learned_model_path

        self.num_vid = num_vid

        ## Initialize W

        Wi_save_dir = os.path.join(directory, "Wi")
        if not os.path.exists(os.path.join(directory, "Wi")):
            os.makedirs(Wi_save_dir)
            print("Initialize Discriminator Weights...")
            for i in trange(self.num_vid):
                w_i_path = os.path.join(Wi_save_dir, "W_{}".format(i))
                if not os.path.exists(w_i_path):
                    os.makedirs(w_i_path)
                if not os.path.exists(os.path.join(w_i_path, "W_{}.pt".format(i))):
                    w_i = torch.rand(512, 1)
                    torch.save({"W_i": w_i}, os.path.join(w_i_path, "W_{}.pt".format(i)))

        self.Wi_save_dir = Wi_save_dir

        self.data_loader, data_length =  load_dataloader(self.dataset, self.path_to_data, self.Wi_save_dir, self.K, self.train_bool, self.finetune_bool, self.batch_size, self.num_vid)
        
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
        
        self.test_video_path = test_video_path
        self.test_model_path =test_model_path
        
        self.ReLU = nn.ReLU()
        self.L1_Norm = nn.L1Loss()

        self.build_model()
        self.build_tensorboard()
        

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

        self.E = Embedder(self.curr_size, self.size, "sum")
        self.G = Generator(self.curr_size, self.size, 5, self.finetune_bool)
        self.D = Discriminator(self.batch_size, self.curr_size, self.size, self.finetune_bool,"sum")

        self.opt_G = torch.optim.Adam([{"params": self.G.parameters()}, {"params": self.E.parameters()}], lr = self.lr_G, betas = (self.beta1, self.beta2))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr = self.lr_D, betas = (self.beta1, self.beta2))
        
        self.print_network(self.E, "Embedder")
        self.print_network(self.G, "Generator")
        self.print_network(self.D, "Discriminator")

        self.E.apply(xavier_init)
        self.G.apply(xavier_init)
        self.D.apply(xavier_init)

        self.E.to(self.device)
        self.G.to(self.device)
        self.D.to(self.device)

        
    def load_model(self, model_path) :
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from {}...'.format(model_path))
        path = model_path
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.epoch = checkpoint['epoch']
        self.restored_batch_idx = checkpoint["batch_idx"]
        self.E.load_state_dict(checkpoint["E"])
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'], strict = False)

        self.opt_G.load_state_dict(checkpoint['opt_G'])
        self.opt_D.load_state_dict(checkpoint['opt_D'])

        self.E.to(self.device)
        self.G.to(self.device)
        self.D.to(self.device)
        # self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)
        
        return

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.opt_G.zero_grad()
        self.opt_D.zero_grad()

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
        return self.d_adv_weight *loss_real, self.d_adv_weight*loss_fake

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
        return self.g_adv_weight * (-1) * torch.mean(fake_score)
    
    def Loss_MCH(self, e_vectors, W):
        # W: 512 , B
        # e_vectors: B, K, 512, 1
        B, K, _ , _ = e_vectors.size ()
        W = W.unsqueeze(-1)         # 512, B, 1
        W = W.expand(512, B, K)     # 512, B, K
        W = W.permute(1, 2, 0).contiguous()    # B, K, 512
        W = W.view(-1, 512)         # B*K, 512

        e_vectors = e_vectors.squeeze(-1)       # B, K, 512
        e_vectors = e_vectors.view(-1, 512)     # B*K, 512
        loss_mch = self.L1_Norm(W, e_vectors)
        return self.g_match_weight * loss_mch
    
    def train(self):

        data_iter = iter(self.data_loader)
        s_img_fixed, s_landmark_fixed, imgs_list_fixed, landmarks_list_fixed,idx_fixed, Wi_fixed = next(data_iter)
        s_img_fixed = s_img_fixed.to(self.device)
        s_landmark_fixed = s_landmark_fixed.to(self.device)
        imgs_list_fixed = imgs_list_fixed.to(self.device)
        landmarks_list_fixed = landmarks_list_fixed.to(self.device)
        B, K, C, H, W = imgs_list_fixed.size()

        imgs_fixed = imgs_list_fixed.view(-1, C, H, W).to(self.device)
        landmarks_fixed = landmarks_list_fixed.view(-1, C, H, W).to(self.device)
        self.epoch = 0
        self.global_step = 0
        if self.resume_epoch > 0:
            self.load_model(self.restored_model_path)
            if self.resume_epoch != self.epoch:
                raise(RuntimeError("Resume epoch should be same with that of loaded model"))
            self.global_step = self.epoch * len(self.data_loader)
            self.epoch = self.resume_epoch

        print('Start training...')
        start_time = time.time()
        while self.epoch <= self.num_epoch:
            for batch_idx, batch_data in enumerate(self.data_loader):
                if batch_idx < self.restored_batch_idx:
                    self.global_step += 1
                    continue
                s_img = batch_data[0].to(self.device)
                s_landmark = batch_data[1].to(self.device) # B, 3, 224, 224 
                imgs_list = batch_data[2].to(self.device)
                landmarks_list = batch_data[3].to(self.device) # B, K , 3, 224, 224
                vid_idx = batch_data[4]
                Wi = batch_data[5].squeeze(-1).permute(1,0).contiguous().to(self.device).requires_grad_()  #B, 512, 1 -> 512, B


                self.D.load_W_i(Wi)

                B, K, C, H, W = imgs_list.size()

                imgs = imgs_list.view(-1, C, H, W).to(self.device)
                landmarks = landmarks_list.view(-1, C, H, W).to(self.device)

                ## train D twice,
                with torch.autograd.set_detect_anomaly(True):
                    with torch.autograd.detect_anomaly():
                        e = self.E(imgs, landmarks) # extract style of imgs , B*K, 512, 1
                        e = e.view(B, K, -1, 1) # B, K, 512, 1
                        e_mean = torch.mean(e, dim = 1)
                        s_fake = self.G(s_landmark, e_mean.detach())
                        
                        ## train D stage 1 
                        real_score, real_disc_feats = self.D(s_img, s_landmark)
                        fake_score, fake_disc_feats = self.D(s_fake.detach(), s_landmark)
                    
                        d_loss_real, d_loss_fake = self.Loss_D(real_score, fake_score)
                        d_loss = (d_loss_real + d_loss_fake)
                        self.reset_grad()
                        d_loss.backward()
                        self.opt_D.step()

                ## train D stage 2

                real_score, real_disc_feats = self.D(s_img, s_landmark)
                fake_score, fake_disc_feats = self.D(s_fake.detach(), s_landmark)

                d_loss_real, d_loss_fake = self.Loss_D(real_score, fake_score)
                d_loss = (d_loss_real + d_loss_fake)
                self.reset_grad()
                d_loss.backward()
                self.opt_D.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_total'] = d_loss.item()

                ## train E,G
                
                s_fake = self.G(s_landmark, e_mean)
                real_score, real_disc_feats = self.D(s_img, s_landmark)
                fake_score, fake_disc_feats = self.D(s_fake, s_landmark)

                g_loss_adv = self.Loss_G_Adv(fake_score)
                g_loss_fm = self.Loss_FM(real_disc_feats, fake_disc_feats)
                g_loss_cnt = self.Loss_CNT(s_img, s_fake)
                g_loss_mch = self.Loss_MCH(e, Wi)
                
                g_loss = g_loss_adv + g_loss_fm + g_loss_cnt + g_loss_mch

                self.reset_grad()
                g_loss.backward()
                self.opt_G.step()

                loss['G/loss_adv'] = g_loss_adv.item()
                loss['G/loss_fm'] = g_loss_fm.item()
                loss['G/loss_cnt'] = g_loss_cnt.item()
                loss['G/loss_mch'] = g_loss_mch.item()
                loss['G/loss_total'] = g_loss.item()

                for enum, idx in enumerate(vid_idx):
                    torch.save({'W_i': self.D.W_i[:,enum].unsqueeze(-1)}, self.Wi_save_dir+'/W_'+str(idx.item())+'/W_'+str(idx.item())+'.pt')
                    

                if self.global_step % self.print_freq == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch[{}/{}], Iteration [{}/{}]".format(et, self.epoch, self.num_epoch, batch_idx, len(self.data_loader))
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, self.global_step)
            
                if self.global_step % self.sample_freq == 0:
                    with torch.no_grad():
                        fake_list = [s_img_fixed]
                        e_fixed= self.E(imgs_fixed, landmarks_fixed) # extract style of imgs , B*K, 512, 1
                        e_fixed = e_fixed.view(B, K, -1, 1) # B, K, 512, 1
                        e_mean_fixed = torch.mean(e_fixed, dim = 1)
                        s_fake = self.G(s_landmark_fixed, e_mean_fixed)
                        fake_list.append(s_fake) # B 
                        
                        fake_concat = torch.cat(fake_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(self.global_step))
                        save_image(self.denorm(fake_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(sample_path))

                if self.global_step % self.model_save_freq == 0:
                    model_path = os.path.join(self.model_save_dir, "{}-checkpoint.pt".format(self.epoch))
                    torch.save({
                        'epoch': self.epoch,
                        'batch_idx': batch_idx,
                        'E': self.E.state_dict(),
                        'G': self.G.state_dict(),
                        'D': self.D.state_dict(),
                        'opt_G': self.opt_G.state_dict(),
                        'opt_D': self.opt_D.state_dict(),
                    }, model_path)
                    # torch.save(self.model.state_dict(), model_path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))
            
                self.global_step += 1

            self.epoch += 1
        


    def finetune(self):
        ## fine tuning stage
        print("Finetuning training init..")
        
        return

    def test(self):
        # video inference
        # Embedding
        embedding_path = os.path.join(self.directory, "embedding.pt")
        self.load_model(self.test_model_path)
        self.G.eval()
        self.E.eval()

        if not os.path.exists(embedding_path):
            T = len(Finetune_Voxceleb2(path_to_data = self.path_to_data, transforms = None, vid_idx= self.num_vid))
            dataloader_for_Emb, data_length = load_dataloader(dataset = self.dataset, path_to_data= self.path_to_data, path_to_Wi= None, K = 0, train = False, finetuning= True, batch_size = T, num_vid=self.num_vid)
            data_iter = iter(dataloader_for_Emb)
            emb_imgs, emb_landmarks = next(data_iter)  # T, 3, 224, 224

            if emb_imgs.size(0) != T:
                raise(RuntimeError("# of emb_imgs are not as same as T"))

            emb_imgs = emb_imgs.to(self.device)
            emb_landmarks = emb_landmarks.to(self.device)
            with torch.no_grad():
                embedding = self.E(emb_imgs, emb_landmarks) # T, 512, 1
                embedding = embedding.view(-1, T, 512, 1) # 1, T, 512, 1
                print("embedding_size before mean", embedding.size())
                embedding = torch.mean(embedding, dim =1 ) # 1, 512, 1
                print(embedding.size())
            
            print("Save embeddings ...")
            torch.save({"embedding": embedding}, embedding_path)
        else:
            embedding_checkpoint = torch.load(embedding_path, map_location=lambda storage, loc: storage)
            embedding = embedding_checkpoint["embedding"].to(self.device)


        print("Start Video Inferencing...")

        cap = cv2.VideoCapture(self.test_video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        ret = True
        i = 0
        size = (256*3,256)
        video = cv2.VideoWriter('sample.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
        with torch.no_grad():
           # for frame_idx in trange(n_frames):
            while ret:
                frame_img, frame_landmark, ret = generate_video_landmarks(cap=cap, device=self.device, pad = 50)
                if ret:
                    frame_img = frame_img.to(self.device)
                    frame_landmark = frame_landmark.to(self.device)

                    frame_img = frame_img.unsqueeze(0)
                    frame_landmark =frame_landmark.unsqueeze(0)

                    embedding = embedding.view(1, 512, 1)
                    self.G.fine_tuning = False

                    fake_img = self.G(frame_landmark, embedding)

                    frame_img = frame_img.squeeze(0)
                    frame_landmark =frame_landmark.squeeze(0)
                    fake_img = fake_img.squeeze(0)

                    ## tensor to pil to cv2
                    fake_img = transforms.Resize(256)(fake_img)
                    fake_img = self.denorm(fake_img.cpu())
                    fake_img_pil = transforms.ToPILImage()(fake_img)
                    fake_img_cv2 = PILtocv2(fake_img_pil)

                    frame_img = self.denorm(frame_img.cpu())
                    frame_img_pil = transforms.ToPILImage()(frame_img)
                    frame_img_cv2 = PILtocv2(frame_img_pil)

                    frame_landmark = self.denorm(frame_landmark.cpu())
                    frame_landmark_pil = transforms.ToPILImage()(frame_landmark)
                    frame_landmark_cv2 = PILtocv2(frame_landmark_pil)
                    
                    #print(fake_img_cv2.shape)
                    #print(frame_img_cv2.shape)
                    #print(frame_landmark_cv2.shape)
                

                    img = np.concatenate((fake_img_cv2, frame_landmark_cv2, frame_img_cv2), axis=1)
                    img = img.astype('uint8')
                    video.write(img)
                
                    i+=1
                    print(i,'/',n_frames)

        print("VIDEO SAVED!")
        cap.release()
        video.release()





#####
# In paper setting ,
# For full dataset(voxceleb2), 150 epochs without fine tuning
# or
# 75 epochs meta learning, 75 epochs for finetuning
#
####### The order for experiment
# 1. For small dataset , experiment whether the model works first
# 2. And then try video inference
# 3. and then finetuning
#######