import os
import shutil

data = '../data'
K = 8


for person_id in os.listdir(data):
    for video_id in os.listdir(os.path.join(data, person_id)):
        for video_num in os.listdir(os.path.join(data, person_id, video_id)):
            check = len(os.listdir(os.path.join(data, person_id, video_id, video_num, 'img')))
            if check != K:
                print("Wrong")
                print(os.path.join(data, person_id, video_id, video_num, 'img'))
                print(check)
                if check == 0:
                    print('remove ', os.path.join(data, person_id, video_id, video_num))
                    shutil.rmtree(os.path.join(data, person_id, video_id, video_num))
                

for person_id in os.listdir(data):
    for video_id in os.listdir(os.path.join(data, person_id)):
        if len(os.listdir(os.path.join(data, person_id, video_id))) ==1 :
            pass
        else:
            print(os.path.join(data, person_id, video_id))
            print(len(os.listdir(os.path.join(data, person_id, video_id))))