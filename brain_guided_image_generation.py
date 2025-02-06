import os
os.chdir('/engram/nklab/hossein/recurrent_models/transformer_brain_encoder/')

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import numpy as np
import torch
# import torchvision.transforms as T
from PIL import Image
import torchvision.transforms as transforms
from datasets.nsd_utils import roi_maps, plot_on_brain
from datasets.nsd import fetch_dataloaders

from brain_encoder_wrapper import brain_encoder_wrapper

import sys
import os
import torch
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from brain_guide_pipeline import mypipelineSAG
import pickle
import gc
import nibabel as nib
import os
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rois_str', help='rois_str', type=str, default='OFA')
args = parser.parse_args()
rois_str = args.rois_str
rois_list = rois_str.split('_')

print(rois_str, rois_list, flush=True)
#################################
# model = brain_encoder_wrapper()
# model_name = 'default'
##################################
subj=1
enc_output_layer=[1, 3, 5, 7]

readout_res= 'rois_all'
runs= np.arange(1,6) 

# readout_res= 'voxels'
# runs= np.arange(1, 3)

model = brain_encoder_wrapper(subj=subj, readout_res=readout_res, enc_output_layer=enc_output_layer, runs=runs)

model_name = 'model'
model_name += 'Voxels' if readout_res == 'voxels' else ''
model_name += 'Layer' + ''.join([str(cur) for cur in enc_output_layer])
model_name += 'Runs' + ''.join([str(cur) for cur in runs])
##################################
print(model_name, flush=True)


if model.model is not None:
    cur_model = model.model
    cur_model.lr_backbone = 1 # otherwise no gradient in brain_encoder line 68
    for name, param in cur_model.named_parameters():
        param.requires_grad = False
else:
    for cur_model in model.models:
        cur_model.lr_backbone = 1
        for name, param in cur_model.named_parameters():
            param.requires_grad = False


repo_id = "stabilityai/stable-diffusion-2-1-base"
pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe2 = pipe.to("cuda")


lh_challenge_rois = []
rh_challenge_rois = []
for roi in rois_list:
    roi_ind = model.lh_roi_names.index(roi)
    lh_challenge_rois.append(model.lh_challenge_rois[roi_ind]) 
    rh_challenge_rois.append(model.rh_challenge_rois[roi_ind])
lh_challenge_rois = torch.clip(torch.stack(lh_challenge_rois).sum(0), min=0, max=1).cpu().numpy()
rh_challenge_rois = torch.clip(torch.stack(rh_challenge_rois).sum(0), min=0, max=1).cpu().numpy()


def loss_function(image_input):
    outputs = model.forward(image_input)
    rois_acts = torch.mean(outputs[0] * torch.tensor(lh_challenge_rois).cuda(), axis=1) 
    rois_acts += torch.mean(outputs[1] * torch.tensor(rh_challenge_rois).cuda(), axis=1)
    return -torch.mean(rois_acts)


fld = '/engram/nklab/ms5724/transformer_brain_encoder/images'
os.makedirs(fld, exist_ok=True)
os.makedirs(f'{fld}/{model_name}', exist_ok=True)
pipe.brain_tweak = loss_function

import time
time_st = time.time()
for seed in range(500):
    
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    
    g = torch.Generator(device="cuda").manual_seed(int(seed))
    image = pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=50, generator=g, clip_guidance_scale=130.0)
    
    image.images[0].save(f'{fld}/{model_name}/{rois_str}_detachk_seed{seed}.png', format="PNG", compress_level=6)
    
    # fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    # ax.imshow(image.images[0])
print(time.time() - time_st) #240/3 #29.9-30.1
