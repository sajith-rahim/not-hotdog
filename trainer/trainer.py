from dataloader.dataloader import ImageNetDataset
from utils.utils import get_device, save_dict
import torch


class Trainer:
    def __init__(self, model, loss, optimizer, train_path):
        super().__init__()
        
        self.loss = loss
        self.train_path = train_path
        self.train_ds = None
        self.device = get_device()
        self.model = model.to(self.device)
        self.optim = optimizer(self.model.parameters(), lr=1e-3)
        

    def load_dataset(self):
        self.train_ds = ImageNetDataset(self.train_path)
        # print(len(self.train_ds))

    def train(self, n_epochs=None):
        n_epochs = 33 if n_epochs is None else n_epochs

        self.load_dataset()


        for epoch in range(n_epochs):
            epoch_loss = dict()
            for idx, items in enumerate(self.train_ds):
                img, label = items

                img = img.unsqueeze(0)
                label = torch.tensor([1.0, 0]) if label == 0 else torch.tensor([0.0, 1.0])
                
                img = img.to(self.device)
                label = label.to(self.device)

                pred = self.model(img)

                _loss = self.loss(pred[0], label)

                self.optim.zero_grad()

                _loss.backward()
                self.optim.step()

                epoch_loss[idx] = _loss.item()
                print(f"[Epoch:{epoch}] {idx} >> Loss: {_loss.item()}")
            print(f"<{self.device}>[Epoch:{epoch}] >> Loss: {torch.tensor(list(epoch_loss.values())).mean()}")
            if epoch%10 == 0:
                save_dict(self.model.state_dict(), 'model_' + str(epoch))
