import wandb
import torch 
from torch import nn
import torch.nn.functional as F

config = wandb.config

class KWSNet(nn.Module):    
    def __init__(self, enc_hidden_size, conv_out_channels, conv_kernel_size):
        super(KWSNet, self).__init__()
        
        self.enc_hidden_size = enc_hidden_size
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
          
        self.conv = nn.Conv1d(config.melspec_n_mels, conv_out_channels, conv_kernel_size)
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
                if idx < (config.window_size - 1):
                    continue
                if idx > (config.window_size  - 1):
                    encoder_outputs = encoder_outputs[1:]
                    
                out = torch.stack(encoder_outputs)

                out = self.attention(out)

                out = self.clf(out)
                ans.append(out)

            return torch.stack(ans) 