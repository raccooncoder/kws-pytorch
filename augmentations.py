import torch
from torch import nn
import torchaudio

youtube_noise, _ = torchaudio.load('Cafe sounds ~ Ambient noise-i9a6ReFTHiw.wav')
youtube_noise = youtube_noise.sum(dim=0)

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0, std=0.05):
        super(GaussianNoise, self).__init__()
        
        self.noiser = torch.distributions.Normal(mean, std)   
            
    def forward(self, wav):
        wav = wav + self.noiser.sample(wav.size())     
        wav = wav.clamp(-1, 1)
        
        return wav
    
class YoutubeNoise(nn.Module):    
    def __init__(self, alpha=0.05):
        super(YoutubeNoise, self).__init__()
        
        self.alpha = alpha
        self.noise_wav = youtube_noise
            
    def forward(self, wav):
        wav = wav + self.alpha * self.noise_wav[:wav.shape[-1]] 
        wav = wav.clamp(-1, 1)
        
        return wav