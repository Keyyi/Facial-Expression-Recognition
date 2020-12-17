# Importing the needed modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from Model import *

class FERDataset(Dataset):

    def __init__(self, images, labels, transforms):
        self.X = images
        self.y = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = [int(m) for m in self.X[i].split(' ')]
        data = np.asarray(data).astype(np.uint8).reshape(48,48,1)
        data = self.transforms(data)
        label = self.y[i]
        return (data, label)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
        
def evaluate(model, val_loader):
# This function will evaluate the model and give back the val acc and loss
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def fit(epochs, max_lr, model, train_loader, val_loader, weight_decay, grad_clip, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []    #keep track of the evaluation results

    # setting upcustom optimizer including weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # setting up 1cycle lr scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # training
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            sched.step()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
    
    
def main():
    print("Get data successfully")
    df = pd.read_csv('fer2013.csv')
    df_train = pd.concat([df[(df.Usage == 'Training')], df[df.Usage == 'PublicTest']], ignore_index=True).drop(['Usage'], axis=1)
    df_test = df[df.Usage == 'PrivateTest'].drop(['Usage'], axis=1).reset_index().drop(['index'], 1)
    train_images = df_train.iloc[:, 1]
    print(type(train_images))
    train_labels = df_train.iloc[:, 0]
    test_images = df_test.iloc[:, 1]
    test_labels = df_test.iloc[:, 0]
    train_trfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(48, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5), inplace=True)
    ])
    val_trfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    print("Initialize train data successfully")
    train_data = FERDataset(train_images, train_labels, train_trfm)
    val_data = FERDataset(test_images, test_labels, val_trfm)
    random_seed = 42
    torch.manual_seed(random_seed)
    
    batch_num = 400
    print("Get train_dl successfully")
    train_dl = DataLoader(train_data, batch_num, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_data, batch_num*2, num_workers=4, pin_memory=True)
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    print("Get model successfully")
    model = to_device(FERModel(1, 7), device)
    print("Begin evalute")
    evaluate(model, val_dl)
    
    max_lr = 0.001
    grad_clip = 0.1
    weight_decay = 1e-4
    print("Begin fit")
    history = fit(30, max_lr, model, train_dl, val_dl, weight_decay, grad_clip, torch.optim.Adam)
    torch.save(model.state_dict(), '9.pth')
    plot_losses(history)
    plt.figure()
    plot_lrs(history)
    
main()
