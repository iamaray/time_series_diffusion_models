import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model_wrappers import DiffusionModel
from .diffusion import DiffusionLoader
import os
import datetime

class SNRGammaMSE:
    def __init__(self, gamma=5):
        self.gamma = gamma
        self.MSE = nn.MSELoss()
        
    def __call__(self, y_pred, y_label, snr):
        return min(self.gamma, snr) * self.MSE(y_pred, y_label)
    
class CRPSLoss(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) loss for probabilistic forecasts.
    Assumes y_pred is a sample or mean prediction, and y_true is the target.
    For a single deterministic prediction, CRPS reduces to MAE.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # y_pred: (batch, ...)
        # y_true: (batch, ...)
        # If y_pred is a sample (batch, n_samples, ...), compute sample CRPS
        if y_pred.dim() > y_true.dim():
            # y_pred: (batch, n_samples, ...)
            y_true = y_true.unsqueeze(1)  # (batch, 1, ...)
            diff1 = torch.abs(y_pred - y_true).mean(dim=1)
            diff2 = torch.abs(y_pred.unsqueeze(2) - y_pred.unsqueeze(1)).mean(dim=(1,2)) * 0.5
            crps = diff1 - diff2
        else:
            crps = torch.abs(y_pred - y_true)
        if self.reduction == 'mean':
            return crps.mean()
        elif self.reduction == 'sum':
            return crps.sum()
        else:
            return crps


class DiffusionTrainer:
    def __init__(
        self,
        model: DiffusionModel,
        forward_process,
        criterion=SNRGammaMSE(),
        post_criterion=nn.MSELoss(),
        lr=0.001,
        use_mixup=False,
        data_prediction=True,
        device=None):
        
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=lr)
        self.data_pred = data_prediction
        self.use_mixup = use_mixup
        self.lr = lr
        
        self.scheduler = None 
        self.forward_proc = forward_process

        self.criterion = criterion
        self.post_criterion = post_criterion

        self.pre_train_history = []
        self.post_train_history = []
        self.post_val_history = []
        
        
    def _train_epoch(self, diffusion_loader):
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        num_batch = 0
        loss = 0.0
        for x, y_lst in diffusion_loader:
            self.optimizer.zero_grad()
            
            y0 = y_lst[0].to(self.device)
            t = torch.randint(1, len(y_lst), (1,),
                              device=self.device).item()

            x = x.to(self.device)
            yt = y_lst[t].to(self.device)
            model_in = yt
            y0 = y_lst[0].to(self.device)
            added_noise = None
            
            if self.use_mixup:
                m = torch.rand_like(yt)
                model_in = (m * yt) + ((1-m) * y0)
            
            outs = self.model(x, model_in, t)
            
            if not self.data_pred:
                added_noise = yt - self.forward_proc.raw_coeffs[t] * y0
                loss += self.criterion(outs, added_noise, self.forward_proc.SNR[t])
            else:
                loss += self.criterion(outs, y0, self.forward_proc.SNR[t])
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            num_batch += 1

        loss.backward()
        self.optimizer.step()
        return loss / num_batch
    
    def _eval_epoch(self, val_loader, forward_process):
        self.model.eval()

        val_loss = 0.0
        num_batches = 0

        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            # print('Here0', x.shape, y.shape)
            preds = self.model.inference(x,  forward_process, None, None)
            loss = self.post_criterion(preds, y)
            val_loss += loss
            num_batches += 1

        return val_loss / num_batches
    
    def train(self, num_epochs, diffusion_loader: DiffusionLoader, val_loader: DataLoader):
        print("STARTING DIFFUSION TRAINING")
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=num_epochs,
            eta_min=self.lr / 100
        )
        
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(diffusion_loader)
            val_loss = self._eval_epoch(val_loader, diffusion_loader.forward_process)
            self.scheduler.step()

            print(
                f"[TRAINING] Epoch {epoch + 1}/{num_epochs} -- Train Loss: {train_loss}, Val Loss {val_loss}")

            self.pre_train_history.append(train_loss)

        save_dir = f"modelsave/diffusion/{self.model.name}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Diffusion training finished, saving model to {save_dir}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(save_dir, f'diffusion_trained_model_{timestamp}.pt'))

        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }