import wandb
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from utils import SoftCrossEntropy

def train(epoch, model, optimizer, error, train_loader, device):
    model.train() 
    model.train_mode()
    
    running_loss = 0.0 

    for i, (images, labels) in enumerate(train_loader):
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
            
            wandb.log({"Loss": running_loss / 50})

            running_loss = 0.0

def train_distill(epoch, teacher_model, student_model, alpha, optimizer, error, train_loader, device):
    student_model.train() 
    student_model.train_mode()
    teacher_model.eval() 
    teacher_model.train_mode()

    running_loss = 0.0 

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = student_model(images)
        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        loss = (1.0 - alpha) * error(outputs, labels) + alpha * SoftCrossEntropy(outputs, teacher_outputs)

        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()

        if (i + 1)% 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (i+ 1) * len(images), len(train_loader.dataset),
                    100. * (i + 1) / len(train_loader), running_loss / 50))
            
            wandb.log({"Loss": running_loss / 50})
            
            running_loss = 0.0
            
def evaluate(model, optimizer, error, val_loader, device):
    model.eval() 
    model.train_mode()

    loss = 0
    correct = 0
    
    ans = []
    gt = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss += F.cross_entropy(outputs, labels, reduction = 'sum').item()

            pred = outputs.data.max(1, keepdim=True)[1]
            
            ans.extend(list(pred.detach().cpu()))
            gt.extend(list(labels.detach().cpu()))
            
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        
    loss /= len(val_loader.dataset)
    
    score = average_precision_score(gt, ans)
    print('\nAverage loss: {:.4f}, AP: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, score, 
        correct, len(val_loader.dataset), 
        100. * correct / len(val_loader.dataset)))

    wandb.log({"Val loss": loss, 
                "Val AP": score})

