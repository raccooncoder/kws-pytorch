import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import json
import glob
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

with open('configs/config.json', 'r') as f:
    config = json.load(f)

wandb.init(config=config, project="kws-dlaudio")
config = wandb.config

from utils import set_seed
from train import train_distill, evaluate
from inference import inference
from dataset import SpeechCommands
from model import KWSNet

set_seed(config.random_seed)

print(device)

paths = []
labels = []

for path in glob.glob('speech_commands/*/*.wav'):
    _, label, _ = path.split('/')
    paths.append(path)
    labels.append(int(label == config.target_class))
    
df = pd.DataFrame({'path': paths, 'label': labels})

X_train, X_test, y_train, y_test = train_test_split(np.array(df['path']), 
                                                    np.array(df['label']), 
                                                    test_size=0.1, 
                                                    stratify=np.array(df['label']),
                                                    random_state=config.random_seed)

train_dataset = SpeechCommands(config, X_train, y_train)
test_dataset = SpeechCommands(config, X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.dataloader_num_workers, pin_memory=True)
val_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.dataloader_num_workers, pin_memory=True)  

student_model = KWSNet(config.enc_hidden_size // 2, config.conv_out_channels, config.conv_kernel_size)
student_model = student_model.to(device)

error = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.lr_scheduler_gamma)

teacher_model = KWSNet(config.enc_hidden_size, config.conv_out_channels, config.conv_kernel_size)
teacher_model.load_state_dict(torch.load('checkpoints/teacher_model.pth'))
teacher_model = teacher_model.to(device)

alpha = config.teacher_alpha   

for epoch in range(config.num_epochs): 
    train_distill(epoch, teacher_model, student_model, alpha, optimizer, error, train_loader, device)
    evaluate(student_model, optimizer, error, val_loader, device)
    lr_scheduler.step()

negative_val = []
positive_val = []

for path, label in zip(X_test, y_test):
    if label == 1:
        positive_val.append(path)
    else:
        negative_val.append(path)


path = positive_val[1]
inference('results/student_positive_example.png', student_model, path, noise=True, device=device)

path = negative_val[1]
inference('results/student_negative_example.png', student_model, path, noise=True, device=device)

torch.save(student_model.state_dict(), 'checkpoints/student_model.pth')
wandb.save('checkpoints/student_model.pth')