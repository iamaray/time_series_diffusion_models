import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .diffusion import DiffusionLoader

class DiffusionTrainer:
    def __init__(
        self,
        model,
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
        
    def _train_epoch(self):
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
            
            outs = self.model(x, model_in)
            
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
    
    def _eval_epoch(self):
        self.model.eval()

        val_loss = 0.0
        num_batches = 0

        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            preds = self.model(x, None)
            loss = self.post_criterion(preds, y)
            val_loss += loss
            num_batches += 1

        return val_loss / num_batches
    
    def train(self, num_epochs, diffusion_loader: DiffusionLoader):
        print("STARTING DIFFUSION TRAINING")
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=num_epochs,
            eta_min=self.lr / 100
        )
        
        for epoch in range(num_epochs):
            train_loss = self._pre_train_epoch(diffusion_loader)

            self.scheduler.step()

            print(
                f"[TRAINING] Epoch {epoch + 1}/{num_epochs} -- Diffusion Loss: {train_loss}")

            self.pre_train_history.append(train_loss)

        save_dir = f"modelsave/diffusion/{self.model.name}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Diffusion training finished, saving model to {save_dir}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(save_dir, f'pretrained_model_{timestamp}.pt'))

        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }