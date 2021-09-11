import os
import sys
sys.path.append('../3rdparty/taming-transformers')
sys.path.append('../3rdparty/CLIP')

import json
import time
import shutil
import imageio
import argparse
import numpy as np
from datetime import datetime

from PIL import Image, ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from utils import *
import clip


VIDEO_IO_PATH = os.environ['VIDEO_IO_PATH']
VIDEO_INPUT_PATH = os.path.join(VIDEO_IO_PATH, "input")
CONFIG_PATH = os.path.join(VIDEO_INPUT_PATH, "config.json")
VIDEO_OUTPUT_PATH = os.path.join(VIDEO_IO_PATH, "output")
VIDEO_FRAME_PATH = os.path.join(VIDEO_OUTPUT_PATH, "frames")
STATUS_FILE = os.path.join(VIDEO_IO_PATH, "STATUS")
FILENAME_PREFIX = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

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
parser.add_argument('-m', '--mode', default='TEST', action='store', type=str, help='Execution mode ["TEST","TEST_FAIL","PROD","SETUP"]')
parser.add_argument('-d', '--test_duration', default=10, action='store', type=int, help='Duration of test run in seconds')
args = parser.parse_args()

if args.mode.upper() not in ["TEST","TEST_FAIL","PROD","SETUP"]:
    raise Exception("Unknown execution mode.")
print(f"Execution mode: {args.mode}")

if args.mode == "SETUP":
    # download & cache models
    load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    sys.exit()

if not os.path.exists(CONFIG_PATH):
    raise Exception("config.json not found.")

with open(CONFIG_PATH) as f:    
    config = json.load(f) 
    print(json.dumps(config, indent=4, sort_keys=True))
    
prompts = config["prompts"]
size = [config["width"], config["height"]]
init_image = os.path.join(VIDEO_INPUT_PATH, config["init_image"])
image_prompts = config["target_images"]
seed = config["seed"]
max_iterations = config["max_iterations"]
uid = config["unique_id"]

step_size = 0.1
init_weight=0.
cutn=64
cut_pow=1.
noise_prompt_seeds = []
noise_prompt_weights = []

VIDEO_FILENAME = f"{FILENAME_PREFIX}-{prompts}".replace(" ", "_")
VIDEO_OUTPUT_ABSPATH = os.path.join(VIDEO_IO_PATH, VIDEO_FILENAME)


if args.mode in ["TEST", "TEST_FAIL"]:
    for i in range(max_iterations):
        print(f"Simulating generating {max_iterations} frames at {args.test_duration/max_iterations:.2f} fps (current {i})")
        write_test_png(size[0], size[1], i, f"{VIDEO_FRAME_PATH}/{i:04}.png")
        with open(STATUS_FILE, "w") as f:
            f.write(f"IN_PROGRESS {uid} FRAME {i}/{max_iterations}")    
        time.sleep(args.test_duration / max_iterations)
        
    if args.mode == "TEST":
        generate_mp4(VIDEO_FRAME_PATH, VIDEO_OUTPUT_PATH)
        print("Completed simulated run.")
        with open(STATUS_FILE, "w") as f:
            f.write(f"COMPLETED {uid} FRAME {max_iterations}/{max_iterations} {VIDEO_FILENAME}.mp4") # TODO write fake video
    if args.mode == "TEST_FAIL":
        with open(STATUS_FILE, "w") as f:
            f.write(f"FAILED {uid}")  
    sys.exit()
    
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
    with open(STATUS_FILE, "w") as f:
        f.write(f"IN_PROGRESS {uid} FRAME {i}/{max_iterations}")
    opt.zero_grad()
    lossAll = ascend_txt()
    losses_str = ', '.join(f'{loss.item():g}' for loss in lossAll)
    print(f'i: {i}, loss: {sum(lossAll).item():g}, losses: {losses_str}')
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
    generate_mp4(VIDEO_FRAME_PATH, VIDEO_OUTPUT_PATH)
    
    with open(STATUS_FILE, "w") as f:
        f.write(f"COMPLETED {uid} FRAME {max_iterations}/{max_iterations} {VIDEO_FILENAME}.mp4")
except:
    with open(STATUS_FILE, "w") as f:
        f.write(f"FAILED {uid}")
    sys.exit()


