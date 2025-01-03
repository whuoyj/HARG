import torch
import torch.nn as nn
import torch.nn.init as init

vidoe_mae_checkpoint = torch.load("/home/ouyangjun/workspace/data/a/shiXiao/code/Masked-Action-Recognition-main/checkpoints/checkpoint.pth")


converted_dict = {"model_state": {}}

for name, p in vidoe_mae_checkpoint["model"].items():
    key = name.replace("encoder.", "backbone.")
    converted_dict['model_state'][key] = p

for name, param in converted_dict['model_state'].items():

    if param.dim() > 1:
        init.xavier_uniform_(param)
    else:
        init.zeros_(param)

# video-mae-k400-1600ep-transferrd

torch.save(converted_dict, "./initial_weight_video-mae-model-converted.pth")


