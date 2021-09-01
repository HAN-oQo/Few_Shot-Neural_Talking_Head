import json
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn

from training.training import Trainer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
CUDA_LAUNCH_BLOCKKING =1
# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main.py <config_path>"))
config_path = sys.argv[1]

# Open config file
with open(config_path) as f:
    config = json.load(f)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    if config["gpu"] == 0:
        device = 'cuda:0'
    if config["gpu"] == 1:
        device = 'cuda:1'
    else:
        device = 'cuda'
else:
    device = 'cpu'

print(device)

if config["path_to_data"] == "":
    raise(RuntimeError("Path to data not specified. Modify path_to_data attribute in config to point to data."))

#
if config["train"] == 1 :
    # Create a folder to store experiment results
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    directory = "{}_{}".format(timestamp, config["id"])
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save config file in experiment directory
    with open(directory + '/config.json', 'w') as f:
        json.dump(config, f)

elif config["train"] == 2:
    directory = config["finetuning"]["meta_learned_path"]
else:
    directory = config["test"]["test_path"]


cudnn.benchmark = True

if config["dataset"] == "Voxceleb2":
    path_to_data = config["path_to_data"]
    resolution = config["resolution"]
    test = config["test"]
    path_to_finetuning_data = config["path_to_finetuning_data"]
    meta_learned_path = config["meta_learned_path"]
    meta_learned_model_path = config["meta_learned_model_path"]

    if config["train"] == 1:
        train = True
        finetuning = False
        training = config["training"]
        batch_size = training["batch_size"]
    elif config["train"] == 2:
        train = True
        finetuning = True
        training = config["finetuning"]
        batch_size = training["batch_size"]
    else:
        train = False
        finetuning = True
        training = config["finetuning"]
        batch_size = test["batch_size"]
else:
    raise(RuntimeError("Requested Dataset unfound"))


trainer = Trainer(device, 
                train= train,
                finetuning = finetuning,
                directory = directory,
                dataset = config["dataset"],
                path_to_data = path_to_data,
                batch_size = batch_size,
                size = resolution,
                path_to_finetuning_data = path_to_finetuning_data,
                meta_learned_path = meta_learned_path,
                meta_learned_model_path = meta_learned_model_path,
                num_vid = training["num_vid"],
                num_epoch = training["num_epoch"],
                resume_epoch = training["resume_epoch"],
                restored_model_path = training["restored_model_path"],
                lr_G = training["lr_G"],
                lr_D = training["lr_D"],
                weight_decay = training["weight_decay"],
                beta1 = training["beta1"],
                beta2 = training["beta2"],
                milestones = training["milestones"],
                scheduler_gamma = training["scheduler_gamma"],
                g_adv_weight = training["g_adv_weight"],
                g_vgg19_weight = training["g_vgg19_weight"],
                g_vggface_weight = training["g_vggface_weight"],
                g_match_weight = training["g_match_weight"],
                g_fm_weight = training["g_fm_weight"],
                d_adv_weight = training["d_adv_weight"],
                print_freq = training["print_freq"],
                sample_freq = training["sample_freq"],
                model_save_freq = training["model_save_freq"],
                test_video_path = test["test_video_path"],
                test_model_path = test["test_model_path"])


if config["train"] == 1:
    trainer.train()

elif config["train"] ==2:
    trainer.finetune()
else:
    trainer.test()
    
