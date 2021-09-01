# preprocessing code is from: https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models/blob/save_disc/dataset/preprocess.py
import torch
import os
from datetime import datetime
import numpy as np
import cv2

from torchvision.utils import save_image
from tqdm import tqdm
import face_alignment
from matplotlib import pyplot as plt
from dataloader.landmark_utils import *
#from .params.params import path_to_mp4, path_to_preprocess

path_to_mp4 = '/ssd/hankyu/talking_head/Few_Shot-Neural_Talking_Head/data/dev/mp4'
path_to_preprocess = './processed_data'

K = 8

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device = 'cuda:0')


if not os.path.exists(path_to_preprocess):
    os.mkdir(path_to_preprocess)


num_vid = 0
for person_id in tqdm(os.listdir(path_to_mp4)):
    for video_id in (os.listdir(os.path.join(path_to_mp4, person_id))):
        for video in os.listdir(os.path.join(path_to_mp4, person_id , video_id)):
            # try:
            video_num = video.split('.')[0]
            video_path = os.path.join(path_to_mp4, person_id, video_id, video)
            pic_path = os.path.join(path_to_preprocess,str(num_vid))
            print(pic_path)
            if not os.path.exists(pic_path):
                os.mkdir(pic_path)
            else:
                print(str(num_vid)+"exists!")
                num_vid +=1
                continue
            
            img_path = os.path.join(pic_path, 'img')
            landmark_path = os.path.join(pic_path, 'landmark')

            if not os.path.exists(img_path):
                os.mkdir(img_path)
            if not os.path.exists(landmark_path):
                os.mkdir(landmark_path)
            
            frame_list = select_frame(video_path, K)
            frame_list = generate_landmarks(frame_list, face_aligner)
            
            if len(frame_list) == K:
                img_list = [frame_list[i][0] for i in range(K)]
                landmark_list = [frame_list[i][1] for i in range(K)]

                for j in range(len(img_list)):
                    img = cv2.cvtColor(img_list[j], cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(img_path , '{}.png'.format(j)), img)
                for k in range(len(landmark_list)):
                    landmark = cv2.cvtColor(landmark_list[k], cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(landmark_path ,  '{}.png'.format(k)), landmark)

            
                num_vid += 1
                
            else:
                print("# of selected frames is wrong!")





print(num_vid)               

            # except:
            #     print("ERROR: ", video_path)
            


