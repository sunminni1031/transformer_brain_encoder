import os
import warnings
import gc
import time
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from brain_encoder_wrapper import brain_encoder_wrapper
from brain_guide_pipeline import mypipelineSAG
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
warnings.filterwarnings('ignore')
os.chdir('/engram/nklab/hossein/recurrent_models/transformer_brain_encoder/')

parser = argparse.ArgumentParser()
parser.add_argument('--rois_str', help='rois_str', type=str, default='OFA')
parser.add_argument('--detach_k', help='detach_k', type=int, default=0)
parser.add_argument('--subj', help='subj', type=int, default=1)
parser.add_argument('--clip_guidance_scale', help='clip_guidance_scale', type=float, default=130)
args = parser.parse_args()
rois_str = args.rois_str
rois_list = rois_str.split('_')
detach_k = bool(args.detach_k)
subj = args.subj
clip_guidance_scale = args.clip_guidance_scale
print(rois_str, rois_list, f'detach_k={detach_k}', f'subj={subj}', flush=True)
print(f'clip_guidance_scale={clip_guidance_scale}', flush=True)
##################################
enc_output_layer = [1, 3, 5, 7]
runs = np.arange(1, 6)
model = brain_encoder_wrapper(subj=subj, enc_output_layer=enc_output_layer, runs=runs, detach_k=detach_k)
model_name = 'model'
model_name += 'Subj' + str(subj)
model_name += 'Layer' + ''.join([str(cur) for cur in enc_output_layer])
model_name += 'Runs' + ''.join([str(cur) for cur in runs])
model_name += 'detachk' if detach_k else ''
print(model_name, flush=True)
##################################
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
##################################
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
##################################

if clip_guidance_scale != 130:
    model_name += '_clipS' + str(int(clip_guidance_scale))
os.makedirs(f'{fld}/{model_name}', exist_ok=True)

repo_id = "stabilityai/stable-diffusion-2-1-base"
pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe2 = pipe.to("cuda")
pipe.brain_tweak = loss_function

time_st = time.time()
seed_list = np.arange(200)
for seed in seed_list:
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    g = torch.Generator(device="cuda").manual_seed(int(seed))
    image = pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=50, generator=g,
                 clip_guidance_scale=clip_guidance_scale)
    image.images[0].save(f'{fld}/{model_name}/{rois_str}_seed{seed}.png', format="PNG", compress_level=6)
print(time.time() - time_st)

###########################################################
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])
mean_act_arr = []
for seed in seed_list:
    image_path = f'{fld}/{model_name}/{rois_str}_seed{seed}.png'
    image = Image.open(image_path)
    img = preprocess(image)
    patch_size = 14
    size_im = (img.shape[0], int(np.ceil(img.shape[1] / patch_size) * patch_size),
               int(np.ceil(img.shape[2] / patch_size) * patch_size),)
    paded = torch.zeros(size_im)
    paded[:, : img.shape[1], : img.shape[2]] = img
    imgs = paded[None, :, :, :]
    with torch.no_grad():
        mean_act = - loss_function(imgs)
        mean_act_arr.append(mean_act.cpu().numpy())
np.save(f'{fld}/{model_name}_{rois_str}_mean_acts.npy', mean_act_arr)

idxs = np.argsort(mean_act_arr)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i in range(5):
    for j in range(2):
        seed = [seed_list[idxs[-(i + 1)]], seed_list[idxs[i]]][j]
        image_path = f'{fld}/{model_name}/{rois_str}_seed{seed}.png'
        image = Image.open(image_path)
        axes[j, i].imshow(image)
        axes[j, i].axis('off')
fig.savefig(f'{fld}/{model_name}_{rois_str}_top_bottom_5.png', bbox_inches='tight')