# Importing the needed modules
import matplotlib.pyplot as plt
import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from model import *
import pandas as pd

### Reference:
### https://medium.com/jovianml/facial-expression-recognition-using-pytorch-b7326ab36157

class Dataset(Dataset):

    def __init__(self, images, labels, transforms):
        self.X = images
        self.y = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]
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
    print(model)
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def fit(epochs, max_lr, model, train_loader, val_loader, weight_decay, grad_clip, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            if grad_clip: nn.utils.clip_grad_value_(model.parameters(), grad_clip)
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
    npzfile = np.load("./read_images/raf_db.npz")
    npzfile1 = np.load("./read_images/toronto_face.npz")
    
    train_images = npzfile["inputs_train"]
    train_labels = np.argmax(npzfile["target_train"], axis=1)
    test_images = npzfile["inputs_valid"]
    test_labels = np.argmax(npzfile["target_valid"], axis=1)
    
    train_images1 = npzfile1["inputs_train"]
    train_labels1 = npzfile1["target_train"]
    test_images1 = npzfile1["inputs_valid"]
    test_labels1 = npzfile1["target_valid"]
    
    label_mapping = {0: 5, 1: 2, 2: 1, 3: 3, 4: 4, 5: 0, 6: 6}
    train_labels = np.vectorize(label_mapping.get)(train_labels)
    test_labels = np.vectorize(label_mapping.get)(test_labels)
    
    train_images = np.concatenate((train_images, train_images1))
    train_labels = np.concatenate((train_labels, train_labels1))
    test_images = np.concatenate((test_images, test_images1))
    test_labels = np.concatenate((test_labels, test_labels1))
    
    train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(48, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5), inplace=True)
    ])
    
    valid_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    
    print("Initialize train data successfully")
    train_data = Dataset(train_images, train_labels, train_transform)
    valid_data = Dataset(test_images, test_labels, valid_transform)
    torch.manual_seed(33)
    batch_num = 120
    print("Get trainDataLoader successfully")
    trainDataLoader = DataLoader(train_data, batch_num, shuffle=True, num_workers=4, pin_memory=True)
    validDataLoader = DataLoader(valid_data, batch_num*2, num_workers=4, pin_memory=True)
    device = get_default_device()
    trainDataLoader = DeviceDataLoader(trainDataLoader, device)
    validDataLoader = DeviceDataLoader(validDataLoader, device)
    print("Get model successfully")
    model = to_device(Model(1, 7), device)
    print("Begin evalute")
    evaluate(model, validDataLoader)
    max_lr = 0.001
    grad_clip = 0.1
    weight_decay = 1e-4
    print("Begin fit")
    trainLog = fit(81, max_lr, model, trainDataLoader, validDataLoader, weight_decay, grad_clip, torch.optim.Adam)
    torch.save(model.state_dict(), 'model.pth')
    plot_losses(trainLog)
    plt.figure()
    plot_lrs(trainLog)
    
if __name__ == "__main__":
    main()
