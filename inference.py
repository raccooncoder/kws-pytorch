import wandb
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import time 

config = wandb.config

def inference(fname, model, path, noise, device):
    model.eval() 
    model.inference_mode()
    
    noise_wav1, _ = torchaudio.load('LJ001-0001.wav')
    noise_wav2, _ = torchaudio.load('LJ001-0014.wav')
    
    wav, sr = torchaudio.load(path)
    wav_proc = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=config.melspec_sample_rate, 
                                                                    n_mels=config.melspec_n_mels, 
                                                                    n_fft=config.melspec_n_fft, 
                                                                    hop_length=config.melspec_hop_length, 
                                                                    f_max=config.melspec_f_max))
    mel_spectrogram = torch.log(wav_proc(wav) + 1e-9)

    if noise:
        noise1_melspec = torch.log(wav_proc(noise_wav1) + 1e-9) 
        noise2_melspec = torch.log(wav_proc(noise_wav2) + 1e-9)
    
        img = torch.cat((noise1_melspec, mel_spectrogram, noise2_melspec), -1)
    else:
        img = mel_spectrogram
    
    img = img.to(device)
    
    start = time.time()
    with torch.no_grad():
        outputs = F.softmax(model(img).squeeze(1), dim=-1).detach().cpu().numpy()[:, 0, 1]

    finish = time.time()
    wandb.log({'Inference time': finish - start})
    
    plt.figure(figsize=(20,10))
    plt.plot(range(len(outputs)), outputs)
    plt.axhline(y=config.confidence_threshold, color='r', linestyle='-')
    wandb.log({fname: wandb.Image(plt)})
    plt.savefig(fname, dpi=500)