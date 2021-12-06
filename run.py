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
from utils.utils import hotdog, nothotdog, print_banner

if __name__ == "__main__":
    task = 'eval'

    if task == 'train':
        dir_path = os.path.dirname(os.path.realpath(__file__))
        train_ds = dir_path + '/dataset/HotdogDataset/train/**/*.JPEG'
        val_ds = dir_path + '/dataset/HotdogDataset/val/**/*.JPEG'
        # train_ds = './HotdogDataset/train/**/*.JPEG';
        train_paths = np.random.choice(glob.glob(train_ds), 28)
        valid_paths = np.random.choice(glob.glob(val_ds), 3)

        # sample, label = next(iter(train_ds))
        # print(sample.shape)

        n_epochs = 256
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
        
        model_tag = 'epoch_model_256'
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
