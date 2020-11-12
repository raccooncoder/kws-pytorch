import glob
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import average_precision_score

import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

target_class = 'marvin'
num_epochs = 20
batch_size = 256
n_cats = 2

#import wandb
#wandb.init(project="kws-dlaudio")

# reproducibility
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
set_seed(13)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(device)

paths = []
labels = []

for path in glob.glob('speech_commands/*/*.wav'):
    _, label, _ = path.split('/')
    paths.append(path)
    labels.append(int(label == target_class))
    
df = pd.DataFrame({'path': paths, 'label': labels})
df.head()

# augmentations

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

IMG_PADDING_LENGTH = 130

class SpeechCommands(Dataset):
    def __init__(self, X, y, train=True):
        self.paths = X
        self.labels = y
        self.train = train
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = torch.zeros(1, 64, IMG_PADDING_LENGTH)
        wav, sr = torchaudio.load(self.paths[idx])
        
        if self.train:
            wav_proc = nn.Sequential(#GaussianNoise(0, 0.01),
                                    YoutubeNoise(0.1),  
                                    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=512, hop_length=128, f_max=4000),
                                    torchaudio.transforms.FrequencyMasking(freq_mask_param=5),
                                    torchaudio.transforms.TimeMasking(time_mask_param=5)
                                    )
        else:
            wav_proc = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=512, hop_length=128, f_max=4000),
                                    )
            
        mel_spectrogram = torch.log(wav_proc(wav) + 1e-9)
        img[0, :, :mel_spectrogram.size(2)] = mel_spectrogram
        
        return img.reshape(64, IMG_PADDING_LENGTH), self.labels[idx]

X_train, X_test, y_train, y_test = train_test_split(np.array(df['path']), 
                                                    np.array(df['label']), 
                                                    test_size=0.1, 
                                                    stratify=np.array(df['label']))

train_dataset = SpeechCommands(X_train, y_train)
test_dataset = SpeechCommands(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

ENC_HIDDEN_SIZE = 128
WINDOW_SIZE = 100
CONV_OUT_CHANNELS = 16
CONV_KERNEL_SIZE = 51
            
class KWSNet(nn.Module):    
    def __init__(self, enc_hidden_size, conv_out_channels, conv_kernel_size):
        super(KWSNet, self).__init__()
        
        self.enc_hidden_size = enc_hidden_size
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
          
        self.conv = nn.Conv1d(64, conv_out_channels, conv_kernel_size)
        self.lstm = nn.GRU(conv_out_channels, enc_hidden_size, bidirectional=True, num_layers=1)
        self.clf = nn.Linear(enc_hidden_size * 2, 2)
        self.attention = lambda x: torch.mean(x, dim=0)
        self.streaming = False
    
    def train_mode(self):
        self.streaming = False
        
    def inference_mode(self):
        self.streaming = True
        
    def forward(self, x):
        
        if self.streaming == False:
            x = F.relu(self.conv(x))
            x = x.permute(2, 0, 1)
            x, _ = self.lstm(x)
            x = self.attention(x)
            x = self.clf(x)
            
            return x     
        
        if self.streaming == True:
            ans = []
            window = []
            encoder_outputs = []
            
            batch_size, input_dim ,seq_len = x.shape
            
            x = x.permute(2, 0, 1)
            hidden = None
            
            for idx, cur_timestamp in enumerate(x):
                window.append(cur_timestamp)
                if idx < (self.conv_kernel_size - 1):
                    continue
                
                if idx > (self.conv_kernel_size - 1):
                    window = window[1:]

                conv_inp = torch.stack(window) # (kernel_size, batch_size, input_dim)
                conv_inp = conv_inp.permute(1, 2, 0) # (batch_size, input_dim, kernel_size)
                out = F.relu(self.conv(conv_inp)) # (batch_size, conv_out_channels, 1 (seq_len))
                out = out.permute(2, 0, 1) # (1 (seq_len), batch_size, conv_out_channels)
                if hidden is None:
                    out, hidden = self.lstm(out)
                else:
                    out, hidden = self.lstm(out, hidden) 
                
                encoder_outputs.append(out)
                if idx < (WINDOW_SIZE - 1):
                    continue
                if idx > (WINDOW_SIZE - 1):
                    encoder_outputs = encoder_outputs[1:]
                    
                out = torch.stack(encoder_outputs)

                out = self.attention(out)

                out = self.clf(out)
                ans.append(out)

            return torch.stack(ans) 

model = KWSNet(ENC_HIDDEN_SIZE, CONV_OUT_CHANNELS, CONV_KERNEL_SIZE)

model = model.to(device)

error = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train(epoch):
    model.train() #don't forget to switch between train and eval!
    model.train_mode()
    
    running_loss = 0.0 #more accurate representation of current loss than loss.item()

    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = error(outputs, labels)

        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()

        if (i + 1)% 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (i+ 1) * len(images), len(train_loader.dataset),
                    100. * (i + 1) / len(train_loader), running_loss / 50))
            
            running_loss = 0.0
            
def evaluate(data_loader):
    model.eval() 
    loss = 0
    correct = 0
    
    ans = []
    gt = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(data_loader)):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss += F.cross_entropy(outputs, labels, reduction = 'sum').item()

            pred = outputs.data.max(1, keepdim=True)[1]
            
            ans.extend(list(pred.detach().cpu()))
            gt.extend(list(labels.detach().cpu()))
            
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        
    loss /= len(data_loader.dataset)
    
    score = average_precision_score(gt, ans)
    print('\nAverage loss: {:.4f}, AP: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, score, 
        correct, len(data_loader.dataset), 
        100. * correct / len(data_loader.dataset)))

for epoch in range(num_epochs): 
    train(epoch)
    evaluate(val_loader)
    lr_scheduler.step()

def inference(path, noise=False):
    model.eval() 
    model.inference_mode()
    
    noise_wav1, _ = torchaudio.load('LJ001-0001.wav')
    noise_wav2, _ = torchaudio.load('LJ001-0014.wav')
    
    wav, sr = torchaudio.load(path)
    wav_proc = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=16000, 
                                                                  n_mels=64, 
                                                                  n_fft=512, 
                                                                  hop_length=128,
                                                                  f_max=4000))
    mel_spectrogram = torch.log(wav_proc(wav) + 1e-9)
    if noise:
        noise1_melspec = torch.log(wav_proc(noise_wav1) + 1e-9) 
        noise2_melspec = torch.log(wav_proc(noise_wav2) + 1e-9)
    
        img = torch.cat((noise1_melspec, mel_spectrogram, noise2_melspec), -1)
    else:
        img = mel_spectrogram
    
    img = img.to(device)
    
    #plt.imshow(img.squeeze(0).detach().cpu().numpy())
    with torch.no_grad():
        outputs = F.softmax(model(img).squeeze(1), dim=-1).detach().cpu().numpy()[:, 0, 1]
    
    plt.figure(figsize=(20,10))
    plt.plot(range(len(outputs)), outputs)
    plt.axhline(y=0.9, color='r', linestyle='-')
    plt.savefig('{}'.format(path))

negative_val = []
positive_val = []

for path, label in zip(X_test, y_test):
    if label == 1:
        positive_val.append(path)
    else:
        negative_val.append(path)

path = positive_val[1]
inference(path, noise=True)

path = negative_val[1]
inference(path, noise=True)