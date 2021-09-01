import os
import random
import torch
from torch.utils import data
from torchvision.datasets.folder import pil_loader
from dataloader.check_img_data import check_img_length

class Finetune_Voxceleb2(data.Dataset):

    def __init__ (self, path_to_data = "/ssd/hankyu/talking_head/Few_Shot-Neural_Talking_Head/processed_data", transforms = None, vid_idx = 1090001):
        # self.path_to_Wi = path_to_Wi
        self.path_to_data = path_to_data
        # self.train = train
        # self.finetuning = finetuning
        self.transforms =  transforms
        # self.K = K
        self.vid_ids = os.listdir(path_to_data)
        self.train_id = []
        self.vid_idx = vid_idx
        self.T = 0
    
        self.prepare()
        print("Finetuning T: ", self.T)
        # print(len(self.vid_data_path))
        # print(self.vid_data_path[0])

    def prepare(self):
        # check_img_length(self.K, self.path_to_data)
        self.vid_data_path= os.path.join(self.path_to_data, str(self.vid_idx))
        if not os.path.exists(self.vid_data_path):
            raise(RuntimeError("Wrong video id for finetuning dataset"))
        print("Check Validity of data..")
        self.img_dir = os.path.join(self.vid_data_path, "img")
        self.landmark_dir = os.path.join(self.vid_data_path, "landmark")
        if not os.path.exists(self.img_dir):
            raise(RuntimeError("Wrong img path for finetuning dataset"))
        if not os.path.exists(self.landmark_dir):
            raise(RuntimeError("Wrong landmark path for finetuning dataset"))
        if len(os.listdir(self.img_dir)) != len(os.listdir(self.landmark_dir)):
            raise(RuntimeError("# of finetuning landmarks should be same as # of finetuning imgs"))
        
        self.T = len(os.listdir(self.img_dir))
                
    
    def __getitem__(self, idx):

        idx = idx % self.T
        specific_img_path = os.path.join(self.img_dir, "{}.png".format(str(idx)))
        specific_landmark_path = os.path.join(self.landmark_dir, "{}.png".format(str(idx)))

        specific_img = pil_loader(specific_img_path)
        specific_landmark = pil_loader(specific_landmark_path)

        if self.transforms is not None:
            specific_img = self.transforms(specific_img)
            specific_landmark = self.transforms(specific_landmark)
        

        return specific_img, specific_landmark

    def __len__(self):
        return (self.T)