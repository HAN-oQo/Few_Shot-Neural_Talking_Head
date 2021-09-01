import os
import random
import torch
from torch.utils import data
from torchvision.datasets.folder import pil_loader
from dataloader.check_img_data import check_img_length
class Voxceleb2(data.Dataset):

    def __init__ (self, path_to_data = "/ssd/hankyu/talking_head/Few_Shot-Neural_Talking_Head/processed_data", path_to_Wi = None, K = 8,  train = True, finetuning = False, transforms = None, num_vid= 1090000):
        self.path_to_Wi = path_to_Wi
        self.path_to_data = path_to_data
        self.train = train
        self.finetuning = finetuning
        self.transforms =  transforms
        self.K = K
        self.vid_ids = os.listdir(path_to_data)
        self.train_id = []
        self.vid_data_path= []
        self.num_vid = num_vid
        self.prepare()

        print(len(self.vid_data_path))
        print(self.vid_data_path[0])

    def prepare(self):
        print("Check Validity of data..")
        check_img_length(self.K, self.path_to_data)

        if self.train:
            if not self.finetuning:
                self.train_ids = self.vid_ids[0:self.num_vid]
            else:
                self.train_ids = self.vid_ids[self.num_vid:]
        else:
            self.train_ids = self.vid_ids[self.num_vid:]

        for id in self.train_ids:
            self.vid_data_path.append(os.path.join(self.path_to_data, id))
        
    
    def __getitem__(self, idx):
        vid_idx = idx
        idx_vid_data_path = self.vid_data_path[idx % len(self.vid_data_path)]
        img_dir = os.path.join(idx_vid_data_path, "img")
        landmark_dir = os.path.join(idx_vid_data_path, "landmark")

        s = random.randint(0, self.K-1)
        specific_img_path = os.path.join(img_dir, "{}.png".format(str(s)))
        specific_landmark_path = os.path.join(landmark_dir, "{}.png".format(str(s)))

        specific_img = pil_loader(specific_img_path)
        specific_landmark = pil_loader(specific_landmark_path)

        if self.transforms is not None:
            specific_img = self.transforms(specific_img)
            specific_landmark = self.transforms(specific_landmark)
        
        img_list = []
        landmark_list = []

        for i in range(self.K):
            img_list.append(pil_loader(os.path.join(img_dir, "{}.png".format(str(i)))))
            landmark_list.append(pil_loader(os.path.join(landmark_dir, "{}.png".format(str(i)))))
        
        img_tensor_list = []
        landmark_tensor_list = []
        
        if self.transforms is not None:
            for i in range(self.K):
                img_tensor_list.append(self.transforms(img_list[i]))
                landmark_tensor_list.append(self.transforms(landmark_list[i]))
        
        imgs = torch.stack(img_tensor_list, dim=0)
        landmarks = torch.stack(landmark_tensor_list, dim = 0)

        if self.path_to_Wi is not None:
            Wi_dir = os.path.join(self.path_to_Wi , "W_{}".format(idx))
            try:
                Wi = torch.load(os.path.join(Wi_dir, "W_{}.pt".format(idx)), map_location=torch.device('cpu'))["W_i"].requires_grad_(False)
            except:
                print("From Dataset, loading W error, index {}".format(idx))
                Wi = torch.rand(512,1)
        else:
            Wi = None

        return specific_img, specific_landmark, imgs, landmarks, vid_idx, Wi 

    def __len__(self):
        return len(self.vid_data_path)