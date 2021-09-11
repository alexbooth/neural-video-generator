import os
import sys
sys.path.append('./taming-transformers')

import json
import time
import shutil
import imageio
import argparse
import numpy as np
from subprocess import Popen, PIPE

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from utils import *
from CLIP import clip


VIDEO_IO_PATH = os.environ['VIDEO_IO_PATH']
VIDEO_INPUT_PATH = os.path.join(VIDEO_IO_PATH, "input")
CONFIG_PATH = os.path.join(VIDEO_INPUT_PATH, "config.json")
VIDEO_OUTPUT_PATH = os.path.join(VIDEO_IO_PATH, "output")
VIDEO_FRAME_PATH = os.path.join(VIDEO_OUTPUT_PATH, "frames")

shutil.rmtree(VIDEO_OUTPUT_PATH, ignore_errors=True)
os.makedirs(VIDEO_OUTPUT_PATH, exist_ok=True)
os.makedirs(VIDEO_FRAME_PATH, exist_ok=True)

# Hardcoding models in for now
model = "vqgan_imagenet_f16_16384"
clip_model = 'ViT-B/32'
vqgan_config = f'{model}.yaml'
vqgan_checkpoint = f'{model}.ckpt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='TEST', action='store', type=str, help='Execution mode [TEST,PROD,SETUP]')
parser.add_argument('-d', '--test_duration', default=10, action='store', type=int, help='Duration of test run in seconds')
args = parser.parse_args()

if args.mode.upper() not in ["TEST","PROD","SETUP"]:
    raise Exception("Unknown execution mode.")
print(f"Execution mode: {args.mode}")

if args.mode == "SETUP":
    # download & cache models
    load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    sys.exit()
    
if args.mode == "TEST": # TODO make this follow the I/O contract and output test video
    interval = args.test_duration/3
    print(f"loading for {interval:.2f}s")
    time.sleep(interval)
    print(f"generating image for {interval:.2f}s")
    time.sleep(interval)
    print(f"generating video for {interval:.2f}s")
    time.sleep(interval)
    print("completed")
    sys.exit()    

if not os.path.exists(CONFIG_PATH):
    raise Exception("config.json not found.")

with open(CONFIG_PATH) as f:    
    config = json.load(f) 
    print(json.dumps(config, indent=4, sort_keys=True))
    
prompts = config["prompts"]
size = [config["width"], config["height"]]
display_freq = config["images_interval"]
init_image = os.path.join(VIDEO_INPUT_PATH, config["init_image"])
image_prompts = config["target_images"]
seed = config["seed"]
max_iterations = config["max_iterations"]
input_images = config["input_images"]
video_filename = config["video_filename"]

step_size = 0.1
init_weight=0.
cutn=64
cut_pow=1.
noise_prompt_seeds = []
noise_prompt_weights = []

# Only vqgan supported now
model_names={
    "vqgan_imagenet_f16_16384": 'ImageNet 16384',
    "vqgan_imagenet_f16_1024":"ImageNet 1024", 
    "wikiart_1024":"WikiArt 1024", 
    "wikiart_16384":"WikiArt 16384", 
    "coco":"COCO-Stuff", 
    "faceshq":"FacesHQ", 
    "sflckr":"S-FLCKR", 
    "ade20k":"ADE20K", 
    "ffhq":"FFHQ", 
    "celebahq":"CelebA-HQ", 
    "gumbel_8192": "Gumbel 8192"
}
                 
name_model = model_names[model]     

if model == "gumbel_8192":
    is_gumbel = True
else:
    is_gumbel = False

if seed == -1:
    seed = None
if init_image == "None":
    init_image = None
elif not os.path.exists(init_image):
    raise Exception("Seed image not found.")
else:
    print(f"Using seed image at {init_image}")

if image_prompts == "None" or not image_prompts:
    image_prompts = []
else:
    image_prompts = image_prompts.split("|")
    image_prompts = [image.strip() for image in image_prompts]

if init_image or image_prompts != []:
    input_images = True

prompts = [frase.strip() for frase in prompts.split("|")]
if prompts == ['']:
    prompts = []

if prompts:
    print('Using prompts:', prompts)
if image_prompts:
    print('Using image prompts:', image_prompts)
if seed is None:
    seed = torch.seed()
torch.manual_seed(seed)
print('Using seed:', seed)

model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

if args.mode == "SETUP": 
    sys.exit()

cut_size = perceptor.visual.input_resolution
if is_gumbel:
    e_dim = model.quantize.embedding_dim
else:
    e_dim = model.quantize.e_dim

f = 2**(model.decoder.num_resolutions - 1)
make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=cut_pow)
if is_gumbel:
    n_toks = model.quantize.n_embed
else:
    n_toks = model.quantize.n_e

toksX, toksY = size[0] // f, size[1] // f
sideX, sideY = toksX * f, toksY * f
if is_gumbel:
    z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
else:
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

if init_image:
    pil_image = Image.open(init_image).convert('RGB')
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
else:
    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
    if is_gumbel:
        z = one_hot @ model.quantize.embed.weight
    else:
        z = one_hot @ model.quantize.embedding.weight
    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
z_orig = z.clone()
z.requires_grad_(True)
opt = optim.Adam([z], lr=step_size)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

pMs = []

for prompt in prompts:
    txt, weight, stop = parse_prompt(prompt)
    embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))

for prompt in image_prompts:
    path, weight, stop = parse_prompt(prompt)
    img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
    batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
    embed = perceptor.encode_image(normalize(batch)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))

for seed, weight in zip(noise_prompt_seeds, noise_prompt_weights):
    gen = torch.Generator().manual_seed(seed)
    embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
    pMs.append(Prompt(embed, weight).to(device))

def synth(z):
    if is_gumbel:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
    else:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

@torch.no_grad()
def checkin(i, losses):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    print(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    #out = synth(z)
    #TF.to_pil_image(out[0].cpu()).save('progress.png')

def ascend_txt():
    global i
    out = synth(z)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    if init_weight:
        result.append(F.mse_loss(z, z_orig) * init_weight / 2)

    for prompt in pMs:
        result.append(prompt(iii))
    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    filename = f"{VIDEO_FRAME_PATH}/{i:04}.png"
    imageio.imwrite(filename, np.array(img))
    return result

def train(i):
    opt.zero_grad()
    lossAll = ascend_txt()
    if i % display_freq == 0:
        checkin(i, lossAll)
    loss = sum(lossAll)
    loss.backward()
    opt.step()
    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))

i = 0
try:
    while True:
        train(i)
        if i == max_iterations:
            break
        i += 1
except KeyboardInterrupt:
    pass

print("Generating video")   
init_frame = 1 #This is the frame where the video will start
last_frame = i #You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.

min_fps = 10
max_fps = 30

total_frames = last_frame-init_frame

length = 15 #Desired video time in seconds

frames = []

for i in range(init_frame,last_frame): #
    filename = f"{VIDEO_FRAME_PATH}/{i:04}.png"
    frames.append(Image.open(filename))

#fps = last_frame/10
fps = np.clip(total_frames/length,min_fps,max_fps)


os.chdir(VIDEO_OUTPUT_PATH)
p = Popen(['ffmpeg', '-y', '-r', '30', '-i', f'{VIDEO_FRAME_PATH}/%04d.png', '-c:v', 'libx264', '-vf', 'fps=30', '-pix_fmt', 'yuv420p', f'{video_filename}.mp4'], stdin=PIPE)
print("The video is now being compressed, wait...")
p.wait()
print("The video is ready")