import torch
import os
from datetime import datetime
import numpy as np
import cv2

from torchvision.utils import save_image
from tqdm import tqdm
import face_alignment
from matplotlib import pyplot as plt
from landmark_utils import *

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device = 'cuda:1')
anime_sample_dir = "../Sample_ANIME"

path_to_data = "/ssd/hankyu/selfie2anime/testB"
idx = 0 
for img in tqdm(os.listdir(path_to_data)):

    pic_path = os.path.join(path_to_data, img)
    pic = cv2.imread(pic_path)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    img_path = os.path.join(anime_sample_dir , "img")
    landmark_path = os.path.join(anime_sample_dir, "landmark")

    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(landmark_path):
        os.makedirs(landmark_path)
    
    frame_list = generate_cropped_landmarks([pic], face_aligner)
    

    if len(frame_list) == 1:
        img_list = [frame_list[i][0] for i in range(1)]
        landmark_list = [frame_list[i][1] for i in range(1)]
        
        img = cv2.cvtColor(img_list[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(img_path , '{}.png'.format(idx)), img)
        
        landmark = cv2.cvtColor(landmark_list[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(landmark_path ,  '{}.png'.format(idx)), landmark)
    else:
        print("Landmark undetected")

    idx += 1 
    
