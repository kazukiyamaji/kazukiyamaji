from pydantic import BaseModel
import torch 
import torch.nn as nn
import base64
from io import BytesIO
import click
import math
import dnnlib
import numpy as np
import PIL.Image as Image
import torch
import random
import copy
import torch.nn.functional as F

import legacy
#GPU
network_pkl="data/network-snapshot-002760.pkl"
with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema']
def pil_to_base64(img,format="jpeg"):
    buffer = BytesIO()
    img.save(buffer,format)
    img_str=base64.b64encode(buffer.getvalue()).decode("ascii")
    return img_str
zigen=256
l0=[]
l1=[]
l2=[]
l3=[]
l4=[]
l5=[]
for i in range(2**5):
    s=[]
    for j in range(5):
        if (i>>j)&1:
            s.append(j)
    eval(f"l{len(s)}").append(s)
i1=0
i2=0
i3=0
i4=0
i5=0
def get_random(values):
    random_alpha1=[]
    for i in range(3):
        random_alpha1.append(values[0]-0.5*i)
        random_alpha1.append(-1*values[0]+0.5*i)
    random_alpha2=[]
    for i in range(3):
        random_alpha2.append(values[1]-0.5*i)
        random_alpha2.append(-1*values[1]+0.5*i)
    random_alpha3=[]
    for i in range(3):
        random_alpha3.append(values[2]-0.5*i)
        random_alpha3.append(-1*values[2]+0.5*i)
    random_alpha4=[]
    for i in range(2):
        random_alpha4.append(values[3]-0.5*i)
        random_alpha4.append(-1*values[3]+0.5*i)
    random_alpha5=[]
    for i in range(2):
        random_alpha5.append(values[4]-0.5*i)
        random_alpha5.append(-1*values[4]+0.5*i)
    return random_alpha1,random_alpha2,random_alpha3,random_alpha4,random_alpha5
random_alpha0_1,random_alpha0_2,random_alpha0_3,random_alpha0_4,random_alpha0_5=get_random([14,9,8,6,6])
random_alpha1_1,random_alpha1_2,random_alpha1_3,random_alpha1_4,random_alpha1_5=get_random([20,15,15,12,12])
random_alpha2_1,random_alpha2_2,random_alpha2_3,random_alpha2_4,random_alpha2_5=get_random([25,20,20,15,15])


def get_dim(layers):
    weights=[]
    num=0
    for i in range(2,int(math.log2(zigen)+1)):
        b=f"b{2**i}"
        if i!=2:
            if num in layers:
                weight=G.synthesis.__getattr__(b).conv0.affine.weight.T
                weights.append(weight.cpu().detach().numpy())
                print(weight.shape)
            num+=1
        
        if num in layers:
            weight=G.synthesis.__getattr__(b).conv1.affine.weight.T
            weights.append(weight.cpu().detach().numpy())
            print(weight.shape)
        num+=1
        if i==int(math.log2(zigen)):
            if num in layers:
                weight=G.synthesis.__getattr__(b).torgb.affine.weight.T
                weights.append(weight.cpu().detach().numpy())
                print(weight.shape)
            num+=1
        
    weight = np.concatenate(weights, axis=1).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values,  boundaries= np.linalg.eig(weight.dot(weight.T))
    return boundaries

layers0=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
boundaries0=get_dim(layers0)
layers1=[0,1,2,3,4,5,6,7,8,9]
boundaries1=get_dim(layers1)
layers2=[10,11,12,13]
boundaries2=get_dim(layers2)

def get_diverse(noise,mode=0,variation=0,direction=0):
    print(mode)
    lays=direction
    sum=0
    i1=random.randint(0,len(l1)-1)
    i2=random.randint(0,len(l2)-1)
    i3=random.randint(0,len(l3)-1)
    i4=random.randint(0,len(l4)-1)
    i5=random.randint(0,len(l5)-1)
    new_codes=noise.detach().clone().repeat(21,1,1).clone()
    layers=eval(f"layers{mode}")
    boundaries=eval(f"boundaries{mode}")
    for i in range(5):
        new_codes[sum,layers,:]+=random.choice(eval(f"random_alpha{variation}_1"))*boundaries[i+lays:i + lays+1]
        sum=(sum+1)%21
        i1=(i1+1)%len(l1)
    for i in range(5):
        for l in l2[i2]:
            new_codes[sum,layers,:]+=random.choice(eval(f"random_alpha{variation}_2"))*boundaries[l+lays:l + 1+lays]
        sum=(sum+1)%21
        i2=(i2+1)%len(l2)

    for i in range(5):
        for l in l3[i3]:
            new_codes[sum,layers,:]+=random.choice(eval(f"random_alpha{variation}_3"))*boundaries[l+lays:l + 1+lays]
        sum=(sum+1)%21
        i3=(i3+1)%len(l3)

    for i in range(5):
        for l in l4[i4]:
            new_codes[sum,layers,:]+=random.choice(eval(f"random_alpha{variation}_4"))*boundaries[l+lays:l + 1+lays]
        sum=(sum+1)%21
        i4=(i4+1)%len(l4)

    for i in range(1):
        for l in l5[i5]:
            new_codes[sum,layers,:]+=random.choice(eval(f"random_alpha{variation}_5"))*boundaries[l+lays:l + 1+lays]
        sum=(sum+1)%21
        i5=(i5+1)%len(l5)
    r=random.sample(range(21), k=9)
    new_codes=new_codes[r]
    return new_codes

#GPUにしたら変更するところ
url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(torch.device('cpu'))

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    Gs = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, Gs.z_dim)
    #GPUにしたら変更するところ
    w_samples = Gs.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in Gs.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, Gs.mapping.num_ws, 1])
        synth_images = Gs.synthesis(ws, noise_mode='const', force_fp32=True)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, Gs.mapping.num_ws, 1])