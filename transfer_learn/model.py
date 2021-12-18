from torchvision.models import vgg16
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils import get_device, save_dict


def imgshow(img, title=None):
    """Imshow for Tensor."""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def train_model(model, loss, optimizer, lr_scheduler, dataloaders, n_epochs = 33):
    device = get_device()
    model = model.to(device)
    best_accuracy = 0.00
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(n_epochs):
        print(f"{epoch}/{n_epochs}")

        for phase in ["train","test"]:

            if phase == "train":
                model.train()
            else:
                model.eval()
        
            loss_ = 0.0
            n_correct = 0

            for imgs, labels in dataloaders[phase]:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    pred = model(imgs)
                    _,_pred_idx = torch.max(pred,1)

                    _loss = loss(pred,labels)

                    #print(labels_vec,pred)
                    
                    if phase == 'train':
                        _loss.backward()
                        optimizer.step()
                    
                    loss_ += _loss.item() * imgs.size(0)
                    n_correct += torch.sum(_pred_idx == labels.data)

                if phase == "train":
                    lr_scheduler.step()
                
            epoch_loss = loss_/len(dataloaders[phase])
            epoch_acc = n_correct.double() / len(dataloaders[phase]) * 10
            print(f" epoch loss: {epoch_loss} -- Accuracy: {epoch_acc}")

        if phase == 'test':
            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
    
    print("Complete")
    save_dict(best_wts, 't_vgg_best_wts')





def transferVGG():
    
    
    device = get_device()

    print("Device:",device)


    mean = [0.485,0.456,0.406]
    std = [0.229,0.224,0.225]

    train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
    ])

    test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
    ])

    transforms_ = {
        'train':train_transforms,
        'test':test_transforms
        }



    train_dataset = ImageFolder(root='dataset/HotdogDataset/train',transform = transforms_['train'])
    train_dataloader = DataLoader(train_dataset,10,shuffle=True,drop_last=True,num_workers=2)

    test_dataset = ImageFolder(root='dataset/HotdogDataset/val',transform = transforms_['test'])
    test_dataloader = DataLoader(test_dataset,10,shuffle=True,drop_last=True,num_workers=2)

    dataloaders = {
        'train': train_dataloader,
        'test' : test_dataloader           
        }


    print(train_dataset.classes)
    #imgs,labels = next(iter(train_dataloader))
    #imgshow(make_grid(imgs),[train_dataset.classes[x] for x in labels])


    model = vgg16(pretrained=True)

    in_ = model.classifier[3].in_features #4096
    out_ = model.classifier[3].out_features #4096
    classfier_layer_dropin_1 = nn.Linear(in_,1024,bias=True)

    classfier_layer_dropin_2 = nn.Linear(1024,2,bias=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[3] = classfier_layer_dropin_1
    model.classifier[6] = classfier_layer_dropin_2



    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=9,gamma=.1)



    train_model(model, loss, optimizer, lr_scheduler, dataloaders, n_epochs = 9)



