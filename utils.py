import torch
import numpy as np
import random 
import torch.nn.functional as F

# reproducibility
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def SoftCrossEntropy(logits, targets):
     return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * F.softmax(targets, dim=1), dim=1))