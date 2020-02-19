import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm_notebook as tqdm
from torch import autograd

from IPython.display import clear_output
from utils import plot_loss_history


def loss(preds, labels, weights):
    """
    Custom weighted BCE with weight masks
    """
    return torch.sum((-1) * (
        weights * labels * torch.log(preds + 1e-8) +\
        weights * (1 - labels) * torch.log(1. - preds + 1e-8)
    )) / preds.shape[0]


def train_model(
    model, metrics,
    train_batch_generator, val_batch_generator,
    opt, lr_scheduler=None,
    ckpt_name=None, n_epochs=30, plot_path=None):
    """
    A function to train a model. While being executed, plots
    the dependency of loss value and metrics from epoch number.
    Saves the parameters of encoder with best loss value in
    checkpoint file  
    :params: 
        model   - a model to be trained. Should be inherited from
                  torch.nn.Module and should implement 'forward()' method

        metrics - callabe to evaluate target metrics on model
                  predictions and correct labels

        train_batch_generator - torch.Dataloader for dataset of
                  CigButtDataset class.

        val_batch_generator   - torch.Dataloader for dataset of 
                  CigButtDataset class.

        opt          - optimizer from torch.optim
        lr_scheduler - scheduler form torch.optim.lr_scheduler 
        cktp_name    - full path to checkpoint file
        n_epochs     - number of epochs, default=30
        save_plots   - a path to save plots of loss and metrics, default=None
    """
    
    device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model   = model.to(device)
    n_train = len(train_batch_generator.dataset)
    n_val   = len(val_batch_generator.dataset)

    batch_size = train_batch_generator.batch_size 

    loss_history_train, metrics_history_train = [], []
    loss_history_val, metrics_history_val     = [], [] 
    

    top_val_metrics_value = 0.0

    total_time = time.time()
    
    for epoch in range(n_epochs):

        train_loss, train_metrics = [], [] # history over the batch
        val_loss, val_metrics     = [], [] # history over the batch
        
        start_time = time.time()

        # Training phase

        model.train(True) 
        for batch in tqdm(train_batch_generator, desc='Training'):
            
            batch_img = batch['image'  ].to(device)
            batch_msk = batch['mask'   ].to(device)
            weights   = batch['weights'].to(device)
            
            opt.zero_grad()
            batch_preds = model.forward(batch_img)
            loss_train = loss(batch_preds, batch_msk, weights)
            with autograd.detect_anomaly():
                try:
                    loss_train.backward()
                except RuntimeError as err:
                    continue

            opt.step()

            train_loss.append(loss_train.cpu().data.numpy())
            train_metrics.append(
                metrics(batch_preds.cpu().data.numpy(),
                        batch_msk.cpu().data.numpy()))
           
            torch.cuda.empty_cache()
        
        # Evaluation phase

        model.train(False)
        for batch in tqdm(val_batch_generator, desc='Validation'):
            
            batch_img = batch['image'  ].to(device)
            batch_msk = batch['mask'   ].to(device)
            weights   = batch['weights'].to(device)
            
            batch_preds = model.forward(batch_img)
            loss_val    = loss(batch_preds, batch_msk, weights)
            
            val_loss.append(loss_val.cpu().data.numpy())
            val_metrics.append(
                metrics(batch_preds.cpu().data.numpy(),
                        batch_msk.cpu().data.numpy()))
           
            torch.cuda.empty_cache()
        
        train_loss_value    = np.mean(train_loss[-n_train // batch_size :])
        train_metrics_value = np.mean(train_metrics[-n_train // batch_size :])
        loss_history_train.append(train_loss_value)
        metrics_history_train.append(train_metrics_value)

        val_loss_value      = np.mean(val_loss[-n_val // batch_size :])
        val_metrics_value   = np.mean(val_metrics[-n_val // batch_size :])
        loss_history_val.append(val_loss_value)
        metrics_history_val.append(val_metrics_value)

        if lr_scheduler: lr_scheduler.step(val_loss_value)
               
        if val_metrics_value > top_val_metrics_value and ckpt_name is not None:
            top_val_metrics_value = val_metrics_value
            with open(ckpt_name, 'wb') as f: torch.save(model, f)
        
        clear_output(True)

        f, axarr = plt.subplots(1, 2, figsize=(16, 8))
        metrics_log = {
            'train': metrics_history_train,
            'val'  : metrics_history_val
        }
        if epoch: plot_loss_history(metrics_log, axarr[1], 'Metrics')
        loss_log = {
            'train': loss_history_train,
            'val'  : loss_history_val
        }
        if epoch: 
            plot_loss_history(loss_log, axarr[0], 'Loss')
            plt.legend()
            if plot_path:
                plt.savefig(os.path.join(plot_path, f'epoch{epoch}.jpg'))
            plt.show() 
            
        # display the results for currrent epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, n_epochs, time.time() - start_time))
        print("  Training metrics: \t{:.6f}".format(train_metrics_value))
        print("  Validation metrics: \t\t\t{:.6f} ".format(val_metrics_value))

    print(f"Trainig took {time.time() - total_time}s in total")
        
    return model, opt, loss_history_val, metrics_history_val
