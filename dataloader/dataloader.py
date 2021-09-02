import random
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms.transforms import CenterCrop
from dataloader.dataset import *
from dataloader.finetune_dataset import *



def load_dataloader(dataset='Voxceleb2', path_to_data= '/ssd/hankyu/talking_head/Few_Shot-Neural_Talking_Head/processed_data', path_to_Wi= None, K=8,  train= True, finetuning = False, batch_size = 16, num_vid = 1090000):
    
    if dataset == 'Voxceleb2':
        
        
        all_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5),
                                std=(0.5,0.5,0.5))
        ])
        if not finetuning:
            Data_Set = Voxceleb2(path_to_data= path_to_data, path_to_Wi = path_to_Wi, K=K, train = train, finetuning = finetuning, transforms = all_transforms, num_vid= num_vid)
            print(len(Data_Set))
            data_loader = data.DataLoader(Data_Set, batch_size = batch_size, shuffle = True,  num_workers = 5, drop_last= True)
            return data_loader, len(Data_Set)
        else:
            Data_Set = Finetune_Voxceleb2(path_to_data = path_to_data, transforms = all_transforms, vid_idx= num_vid)
            print(len(Data_Set))
            data_loader = data.DataLoader(Data_Set, batch_size = batch_size, shuffle = False,  num_workers = 1, drop_last= True)
            return data_loader, len(Data_Set)

    else:
        raise(RuntimeError("Wrong Dataset"))