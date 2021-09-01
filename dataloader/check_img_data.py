import os
import shutil
from tqdm import tqdm

data_path = '../processed_data'
K = 8

def check_img_length(K, data_path):
    for id in tqdm(os.listdir(data_path)):
        vid_id_path = os.path.join(data_path, id)
        img_path = os.path.join(vid_id_path, "img")
        landmark_path = os.path.join(vid_id_path, "landmark")

        if len(os.listdir(img_path)) != K:
            print("wrong image length!")
            print(img_path)
        if len(os.listdir(landmark_path)) != K:
            print("wrong image length!")
            print(landmark_path)

# check_img_length(K, data_path)