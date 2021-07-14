import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
# module_path="D:\\Tony\\SRTP\\WheatMappingModel"
sys.path.append("D:\\Tony\\SRTP\\WheatMappingModel") #需要定义下module path
from Model.wmm import WMM
from timer import record_time

class WheatMappingDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {"sample_x":self.x[idx], "sample_y": self.y[idx]}

class WMMhelper():
    def __init__(
        self, batch_size = 2, input_feature_size=2, lr=0.0005, weight_decay=0.001, factor=0.5, patience=5, max_epoch = 100,
        hidden_dim = 128, kernelsize=(3,3), num_layers=4
    ):

        self.dataset_size = None

        self.batch_size = batch_size

        self.input_feature_size =input_feature_size
        
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.lr = lr
        self.weight_decay=weight_decay

        self.max_epoch = max_epoch
        # scheduler
        self.factor = factor
        self.patience = patience

        self.train_time_list = []
        self.test_time_list = []

        self.hidden_dim = hidden_dim
        self.kernel_size = kernelsize
        self.num_layers=num_layers
        


    def _input_data(self, paths):
        data = np.stack([np.load(path) for path in paths], axis=0)
        return data
    # x shape (b, t, c, h, w)
    def input_x(self, paths):
        """
        input x should be of size (timestep, channels, height, width)
        return ndarray (num of pic, timestep, channels, height, width) dtype = float32

        """
        x = self._input_data(paths).astype("float32")
        self.dataset_size = x.shape[0]
        return x
    
    def input_y(self, paths):
        """
        input y should be of size (height, width)
        return ndarray: (num of pic, height, width) dtype = int64
        """
        y = self._input_data(paths).astype("int64")
        return y
    
    def normalize(self, x_train, x_test):
        """
        input size:
            x_train:(num of pic, timestep, channels, height, width)
            x_test:(1, timestep, channels, height, width)
        output_size:
            the same as input
        """
        scaler = torch.nn.BatchNorm3d(
            self.input_feature_size, affine=False,
        ) 
        scaler.train()
        # x_train after transpose (num of pic, channels, timestep, height, width) 
        # perform batch norm on channels dim
        x_train = scaler(
            torch.FloatTensor(x_train.transpose((0, 2, 1, 3, 4)))
        ).numpy().transpose((0, 2, 1, 3, 4))
        scaler.eval()
        x_test = scaler(
            torch.FloatTensor(x_test.transpose((0, 2, 1, 3, 4)))
        ).numpy().transpose((0, 2, 1, 3, 4))
        return x_train, x_test
    
    def _collate_fn(self, batch):
        return {
            "x": torch.FloatTensor(np.array([sample["sample_x"] for sample in batch])),
            "y": torch.FloatTensor(np.array([sample["sample_y"] for sample in batch]))
        }

    def make_dataloader(self, x, y):
        return DataLoader(WheatMappingDataset(x, y), batch_size=self.batch_size, shuffle=True, num_workers=2, collate_fn=self._collate_fn, drop_last=False)
    
    
    def build_model(self):
        """
        return a WMM model
        """
        return WMM(
            input_dim = self.input_feature_size, hidden_dim = self.hidden_dim, kernel_size = self.kernel_size, num_layers = self.num_layers
        )

    def _eval_perf(self, net, dataloader, device):
        net.eval()
        with torch.no_grad():
            losses = 0
            correct = 0
            total_pixel = 0
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i,batch in enumerate(dataloader):
                xt, yt = batch["x"].to(device), batch["y"].to(device)
                output, _ = net(xt)
                #output size: (batch, height, width)
                loss =self.criterion(output, yt)
                losses += loss.item()
                m = nn.Sigmoid()
                outputs = m(output)
                outputs[outputs > 0.5] = 1
                outputs[outputs <= 0.5] = 0

                correct += (outputs == yt).sum().item()
                TP += (outputs*yt).sum().item()
                FP += ((1-outputs)*yt).sum().item()
                FN += (outputs*(1-yt)).sum().item()
                TN += ((1-outputs)*(1-yt)).sum().item()
                total_pixel += yt.numel()
                # if(TP!=correct):
                #     print(TP)
                #     print(correct)
            # print(output)
            # print(outputs)
            # print(yt)
            running_loss = losses / (len(dataloader.dataset))
            acc = correct / total_pixel
            precision = TP / (TP + FP + 1e-7)
            recall = TP / (TP + FN + 1e-7)
            F1 = 2*(precision*recall) / (precision + recall + 1e-7)
            print(("TP={:4f} FP={:4f} FN={:4f} TN={:4f}, correct={:4f}").format(TP, FP, FN, correct-TP,correct))
        net.train()
        return running_loss , acc, F1
    
    def _train_model(self, net, train_dataloader, test_dataloader, device, logger):
        optimizer = optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.factor, patience=self.patience, verbose=True
        )
        for epoch in range(1, self.max_epoch+1):
            net.train()
            for i,batch in enumerate(train_dataloader):
                xt_train_batch = batch["x"].to(device)
                yt_train_batch = batch["y"].to(device)
                optimizer.zero_grad()
                outputs, _ = net(xt_train_batch)
                loss = self.criterion(outputs, yt_train_batch)
                loss.backward()
                optimizer.step()
            loss_train, acc_train, F1_train = self._eval_perf(
                net, train_dataloader, device
            )
            scheduler.step(loss_train)
            loss_test, acc_test, F1_test =self._eval_perf(
                net, test_dataloader, device
            )
            logger.info((
                "[epoch {:d}]"
                "training loss: {:.10f}, test loss: {:.10f}, F1 score: {:.8f} "
                "training acc: {:.8f}, test acc: {:.8f}, F1 score: {:.8f} "
                " (lr => {:f})"
            ).format(
                epoch,
                loss_train, loss_test, F1_train,
                acc_train, acc_test, F1_test,
                optimizer.param_groups[0]["lr"])
            )
        logger.info("Training completed!")
    
    def train_model(
        self, net, train_dataloader, test_dataloader, device, logger
    ):
        record_time(
            self.train_time_list, self._train_model, [net, train_dataloader, test_dataloader, device, logger]
        )
    
    def _predict(self, net, dataloader, device):
        net.eval()
        y_pred_list = []
        with torch.no_grad():
            for i,batch in enumerate(dataloader):
                xt = batch["x"].to(device)
                outputs, _ = net(xt)
                m = nn.Sigmoid()
                y_pred = m(outputs)
                y_pred[y_pred>0.5] = 1
                y_pred[y_pred<=0.5] = 0
                y_pred_list.append(y_pred)
            y_pred_list = torch.cat(y_pred_list, dim=0).cpu().numpy()
            return y_pred_list

    def predict(self, net, dataloader, device):
        return record_time(
            self.test_time_list, self._predict, [
                net, dataloader, device
            ]
        )



            
            