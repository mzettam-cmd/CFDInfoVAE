"""
functions for training

@yuningw
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np



def compute_mmd(z, prior_z=None):
    if prior_z is None:
        prior_z = torch.randn_like(z)

    def gaussian_kernel(x, y, sigma=1.0):
        return torch.exp(-((x - y)**2).mean() / (2 * sigma**2))

    return (
        gaussian_kernel(z, z) +
        gaussian_kernel(prior_z, prior_z) -
        2 * gaussian_kernel(z, prior_z)
    )





# =========================================================
# TRAIN
# =========================================================
def train_epoch_infovae(model, data, optimizer, alpha, lam, device):
    
    epoch_start = time.time()
    logVar_batch = [] # Store batch logVar to count collapsed modes
    model.train()

    total_loss, total_mse, total_kld, total_mmd = 0, 0, 0, 0
    n = 0

    for x in data:

        x = x.to(device)
        batch_size = x.size(0)

        recon, mu, logvar, z = model(x)

        # -------------------------
        # Reconstruction
        # -------------------------
        mse = F.mse_loss(recon, x, reduction='mean')

        # -------------------------
        # KL
        # -------------------------
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        #kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1)
        #kld = kld.mean()

        # -------------------------
        # MMD (NORMALISÉ)
        # -------------------------
        mmd = compute_mmd(z) / batch_size
        #mmd = compute_mmd(z)

        # -------------------------
        # LOSS InfoVAE
        # -------------------------
        loss = mse + alpha * kld + lam * mmd
        #loss = (mse + (1 - alpha) * kld + (lam + alpha - 1) * mmd)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------------------------
        # accumulation (proper)
        # -------------------------
        total_loss += loss.item() * batch_size
        total_mse  += mse.item() * batch_size
        total_kld  += kld.item() * batch_size
        total_mmd  += mmd.item() * batch_size
        logVar_batch.append(np.exp(0.5* np.mean(logvar.detach().cpu().numpy(), 0)))

        n += batch_size
    train_time = time.time() - epoch_start
    return total_loss / n, total_mse / n, total_kld / n, total_mmd / n, train_time, (np.mean(np.stack(logVar_batch, axis=0), 0) < 0.1).sum() # count collapsed modes


# =========================================================
# TEST
# =========================================================
def test_epoch_infovae(model, data, alpha, lam, device):
    
    epoch_start = time.time()
    model.eval()

    total_loss, total_mse, total_kld, total_mmd = 0, 0, 0, 0
    n = 0

    with torch.no_grad():

        for x in data:

            x = x.to(device)
            batch_size = x.size(0)

            recon, mu, logvar, z = model(x)

            mse = F.mse_loss(recon, x, reduction='mean')
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            #kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1)
            #kld = kld.mean()

            mmd = compute_mmd(z) / batch_size
            #mmd = compute_mmd(z)

            loss = mse + alpha * kld + lam * mmd
            #loss = (mse + (1 - alpha) * kld + (lam + alpha - 1) * mmd)

            total_loss += loss.item() * batch_size
            total_mse  += mse.item() * batch_size
            total_kld  += kld.item() * batch_size
            total_mmd  += mmd.item() * batch_size
            

            n += batch_size
    train_time = time.time() - epoch_start
    return total_loss / n, total_mse / n, total_kld / n, total_mmd / n, train_time

def gaussian_noise(x, var, device):
    
    """
    
    Add gaussian noise on std term for reprarameterization



    """
    return torch.normal(0.0, var, size=x.shape)



def printProgress(epoch, epochs, loss, loss_test, MSE, KLD, elapsed, elapsed_test, collapsed):
    print(f"Epoch: {epoch:3d}/{epochs:d}, Loss: {loss:2.4f}, Loss_test: {loss_test:2.4f}, MSE: {MSE:2.4f}, KLD: {KLD:2.4f}, collapsed: {collapsed:2d}, time train: {elapsed:2.3f}, time test: {elapsed_test:2.3f}")

##############################################
# Functions used for training beta-VAE
################################################


##############################################
# Functions used for training temporal-dynamics predictor
################################################


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def fitting(device,
        model,
        dl,
        loss_fn,
        Epoch,
        optimizer:torch.optim.Optimizer, 
        val_dl        = None,
        scheduler:list= None,
        if_early_stop = True,patience = 10,
        ):
    
    """
    A function for training loop

    Args: 
        device      :       the device for training, which should match the model
        
        model       :       The model to be trained
        
        dl          :       A dataloader for training
        
        loss_fn     :       Loss function
        
        Epochs      :       Number of epochs 
        
        optimizer   :       The optimizer object
        
        val_dl      :       The data for validation
        
        scheduler   :       A list of traning scheduler
        

    Returns:
        history: A dict contains training loss and validation loss (if have)

    """

    from tqdm import tqdm
    
    history = {}
    history["train_loss"] = []
    
    if val_dl:
        history["val_loss"] = []
    
    model.to(device)
    print(f"INFO: The model is assigned to device: {device} ")

    if scheduler is not None:
        print(f"INFO: The following schedulers are going to be used:")
        for sch in scheduler:
            print(f"{sch.__class__}")

    print(f"INFO: Training start")

    if if_early_stop:
        early_stopper = EarlyStopper(patience=patience,min_delta=0)
        print("INFO: Early-Stopper prepared")

    start_epoch = 0

    if os.path.exists("checkpoint_latest.pth"):
        checkpoint = torch.load("checkpoint_latest.pth", map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            for sch, sch_state in zip(scheduler, checkpoint["scheduler_state_dict"]):
                sch.load_state_dict(sch_state)

        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1

        print(f"INFO: Resume training from epoch {start_epoch}")

    for epoch in range(start_epoch, Epoch):
        #####
        #Training step
        #####
        model.train()
        loss_val = 0; num_batch = 0
        for batch in tqdm(dl):
            x, y = batch
            x = x.to(device).float(); y =y.to(device).float()
            optimizer.zero_grad()
            
            pred = model(x)
            loss = loss_fn(pred,y)
            loss.backward()
            optimizer.step()

            

            loss_val += loss.item()/x.shape[0]
            num_batch += 1

        history["train_loss"].append(loss_val/num_batch)

        if scheduler is not None:
            lr_now = 0 
            for sch in scheduler:
                sch.step()
                lr_now = sch.get_last_lr()
            print(f"INFO: Scheduler updated, LR = {lr_now} ")

        if val_dl:
        #####
        #Valdation step
        #####
            loss_val = 0 ; num_batch = 0 
            model.eval()
            for batch in (val_dl):
                x, y = batch
                x = x.to(device).float(); y =y.to(device).float()
                pred = model(x)
                loss = loss_fn(pred,y)
            
                loss_val += loss.item()/x.shape[0]
                num_batch += 1

            history["val_loss"].append(loss_val/num_batch)
        
        train_loss = history["train_loss"][-1]
        val_loss = history["val_loss"][-1]
        print(
                f"At Epoch    = {epoch},\n"+\
                f"Train_loss  = {train_loss}\n"+\
                f"Val_loss    = {val_loss}\n"          
            )
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": [sch.state_dict() for sch in scheduler] if scheduler is not None else None,
            "history": history,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, "checkpoint_latest.pth")
        
        
        if if_early_stop:
            if early_stopper.early_stop(loss_val/num_batch):
                print("Early-stopp Triggered, Going to stop the training")
                break
    return history





