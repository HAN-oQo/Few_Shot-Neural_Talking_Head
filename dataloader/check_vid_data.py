import os
from tqdm import tqdm

#1092009

path_to_mp4 = '../data/dev/mp4'

num = 0

for person_id in tqdm(os.listdir(path_to_mp4)):
    for video_id in (os.listdir(os.path.join(path_to_mp4, person_id))):
        num += len( os.listdir(os.path.join(path_to_mp4, person_id , video_id)))

print(num)
            