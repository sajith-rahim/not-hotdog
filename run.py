import numpy as np;

import glob
import torch
import torch.nn as nn
import os
import cv2

from dataloader.dataloader import ImageNetDataset
from eval.eval import HotDogNotHotDogInfer
from model.HotDogModel import HotDogModel
from trainer.trainer import Trainer
from utils.utils import hotdog, nothotdog, print_banner,get_device


if __name__ == "__main__":
    task = 'train'

    if task == 'train' or task == 'val':
        dir_path = os.path.dirname(os.path.realpath(__file__))
       

        if task == 'train':    
            train_ds = dir_path + '/dataset/HotdogDataset/train/**/*.jpg' 
            # train_ds = './HotdogDataset/train/**/*.JPEG';
            train_paths = np.random.choice(glob.glob(train_ds), 277)

            # sample, label = next(iter(train_ds))
            # print(sample.shape)

            n_epochs = 33
            model = HotDogModel()
            loss = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adagrad

            trainer = Trainer(
                model=model,
                loss=loss,
                optimizer=optimizer,
                train_path=train_paths
            )

            trainer.train(n_epochs)
        else:
            val_ds = dir_path + '/dataset/HotdogDataset/val/**/*.jpg'
            valid_paths = np.random.choice(glob.glob(val_ds), 100)

            valid_ds = ImageNetDataset(valid_paths)

            _pred = []
            _label = []
            device = get_device()

            model_tag = 'epoch_model_30'
            model = HotDogNotHotDogInfer()
            model.load_model(model_tag)
            model.set_device(device)
            for idx, items in enumerate(valid_ds):
                img, label = items

                img = img.unsqueeze(0)
                
                _label.append(label.item())

                img = img.to(device)
                

                pred = model.infer(img)
                pred = torch.argmax(pred).item()

                _pred.append(pred)
                
            print(_pred)
            _pred = torch.tensor(_pred,dtype=torch.float64)
            _label = torch.tensor(_label,dtype=torch.float64)
            with torch.no_grad():
                correct = 0
                correct += torch.sum(_pred == _label).item()
            acc = correct / len(_label)
            print(f"Accuracy : {acc*100}%")

    else:
        
        model_tag = 'epoch_model_30'
        infer_model = HotDogNotHotDogInfer()
        infer_model.load_model(model_tag)
        
        
            
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/dataset/HotdogDataset/val/**/*.JPEG'
        path = np.random.choice(glob.glob(path), 1)[0]
        query = ImageNetDataset.load_img(path)

        prediction = infer_model.infer(query)

        labels = {0 : 'not hotdog', 1: 'hotdog'}

        print_banner()
        if torch.argmax(prediction).item() == 0:
            nothotdog()
        else:
            hotdog()
        print(f"Source: {path}")
        print(labels[torch.argmax(prediction).item()])

        #accuracy_dict = dict()
        #count = 0.0
        #for idx in range(256):

        #    for i in range(3):

        #        if 'nothotdog' in path and labels[torch.argmax(prediction).item()] == 'not hotdog':
        #           count = count + 1

        #    acc_dict[idx] = count/3
        #    count = 0.0

        #print(acc_dict)
