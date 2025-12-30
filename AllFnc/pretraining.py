from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import copy
import datetime
from itertools import zip_longest
import os
import sys
import math
import random
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from selfeeg.ssl.base import EarlyStopping, SSLBase, evaluate_loss



class VICReg(SSLBase):
    """
    See selfeeg vicreg for more info.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: Union[list[int], nn.Module],
    ):

        super(VICReg, self).__init__(encoder)
        self.encoder = encoder
        self._sslname = "vicreg"

        if isinstance(projection_head, list):
            if len(projection_head) < 2:
                raise ValueError("got a list with only one element")
            else:
                if all(isinstance(i, int) for i in projection_head):
                    DenseList = []
                    for i in range(len(projection_head) - 1):
                        DenseList.append(nn.Linear(projection_head[i], projection_head[i + 1]))
                        if i < (len(projection_head) - 2):
                            DenseList.append(nn.BatchNorm1d(num_features=projection_head[i + 1]))
                            DenseList.append(nn.ReLU())    
                    self.projection_head = nn.Sequential(*DenseList)
                else:
                    raise ValueError("got a list with non integer values")
        else:
            self.projection_head = projection_head

    def forward(self, x, mask=None):
        # added mask for transformer pretraining
        if mask is not None:
            x = self.encoder(x, mask)
        else:
            x = self.encoder(x)
        emb = self.projection_head(x)
        return emb

    def get_encoder(self, device="cpu", as_ordered_dict=False):
        """
        Returns a copy of the encoder on the selected device.

        Parameters
        ----------
        device: torch.device or str, optional
            The pytorch device where the encoder must be moved.

            Default = 'cpu'
        
        as_ordered_dict: bool, optional
            If True, returns the encoder weights as an OrderedDict, which can be
            passed to the ``torch.nn.Module.load_state_dict()`` method.

        """
        if as_ordered_dict:
            enc = OrderedDict(
                [(k, v.to(device=device, copy=True)) for k, v in self.encoder.state_dict().items()]
            )
        else:
            enc = copy.deepcopy(self.encoder).to(device=device)
        return enc
        


def pretrain_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    epochs=1,
    optimizer=None,
    preaugmenter=None,
    augmenter=None,
    loss_func: Callable = None,
    loss_args: list or dict = [],
    validation_loss_func: Callable = None,
    validation_loss_args: list or dict = [],
    label_encoder: Callable or list[Callable] = None,
    lr_scheduler=None,
    EarlyStopper=None,
    validation_dataloader: torch.utils.data.DataLoader = None,
    verbose=True,
    device: str or torch.device = None,
    return_loss_info: bool = False,
    mask_tokens: bool= False,
    both_mask_and_aug: bool=False,
    mask_percentage: float or Callable=0.2,
    token_num: int = 498
) -> Optional[dict]:
    """
    read selfeeg.ssl.base.fine_tune help
    """

    if device is None:
        device = torch.device("cpu")
    else:
        if isinstance(device, str):
            device = torch.device(device.lower())
        elif isinstance(device, torch.device):
            pass
        else:
            raise ValueError("device must be a string or a torch.device instance")
    model.to(device=device)

    if not (isinstance(train_dataloader, torch.utils.data.DataLoader)):
        raise ValueError("train_dataloader must be a pytorch DataLoader")
    if not (isinstance(epochs, int)):
        epochs = int(epochs)
    if epochs < 1:
        raise ValueError("epochs must be bigger than 1")
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    if loss_func is None:
        raise ValueError("loss function not given")
    if not (isinstance(loss_args, list) or isinstance(loss_args, dict)):
        raise ValueError("loss_args must be a list or a dict")

    perform_validation = False
    if validation_dataloader != None:
        if not (isinstance(validation_dataloader, torch.utils.data.DataLoader)):
            raise ValueError("validation_dataloader must be a pytorch DataLoader")
        else:
            perform_validation = True
            if validation_loss_func is None:
                validation_loss_func = loss_func
                validation_loss_args = loss_args

    if EarlyStopper is not None:
        if EarlyStopper.monitored == "validation" and not (perform_validation):
            print(
                "Early stopper monitoring is set to validation loss,"
                "but no validation data are given. "
                "Internally changing monitoring to training loss"
            )
            EarlyStopper.monitored = "train"

    # added part
    augment_input = True
    if (augmenter is None) and not(mask_tokens) and not(both_mask_and_aug):
        raise ValueError("set mask_tokens to True or pass a data augmenter")
    if both_mask_and_aug and (augmenter is None):
        raise ValueError("both_mask_and_aug is True but no augmenter was passed")
    if mask_tokens or both_mask_and_aug:
        mask = torch.zeros(token_num, token_num, device=device, dtype = torch.bool)
        token_to_mask = torch.zeros(token_num, device=device, dtype=torch.bool)
        if not(callable(mask_percentage)):
            token_to_mask[:int(token_num*mask_percentage)] = True
        
    loss_info = {i: [None, None] for i in range(epochs)}
    N_train = len(train_dataloader)
    N_val = 0 if validation_dataloader is None else len(validation_dataloader)
    for epoch in range(epochs):
        print(f"epoch [{epoch+1:6>}/{epochs:6>}]") if verbose else None

        train_loss = 0
        val_loss = 0
        train_loss_tot = 0
        val_loss_tot = 0

        # new part - if dynamic percentage of masked token is passed, update it
        if callable(mask_percentage):
            token_to_mask = torch.zeros(token_num, device=device, dtype=torch.bool)
            curr_percentage = mask_percentage(epochs)
            token_to_mask[:int(token_num*curr_percentage)] = True

        if not (model.training):
            model.train()
        with tqdm.tqdm(
            total=N_train + N_val,
            ncols=100,
            bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
            " [{rate_fmt}{postfix}]",
            disable=not (verbose),
            unit=" Batch",
            file=sys.stdout,
        ) as pbar:
            for batch_idx, X in enumerate(train_dataloader):

                optimizer.zero_grad()

                X = X.to(device=device)
                if preaugmenter is not None:
                    X = preaugmenter(X)

                # masking of tokens is applied in the forward using pytorch
                # transformerEncoder "mask" argument 
                if both_mask_and_aug:
                    shuffle1 = torch.randperm(token_num)
                    shuffle2 = torch.randperm(token_num)
                    mask1 = torch.logical_or(
                        torch.logical_or(mask, token_to_mask[shuffle1]),
                        token_to_mask[shuffle1].unsqueeze(1)
                    )
                    mask2 = torch.logical_or(
                        torch.logical_or(mask, token_to_mask[shuffle2]),
                        token_to_mask[shuffle2].unsqueeze(1)
                    )
                    
                    data_aug_1 = augmenter(X)
                    data_aug_2 = augmenter(X)
                    
                    projection_1 = model(data_aug_1, mask1)
                    projection_2 = model(data_aug_2, mask2)
                elif mask_tokens:
    
                    shuffle1 = torch.randperm(token_num)
                    shuffle2 = torch.randperm(token_num)
                    mask1 = torch.logical_or(
                        torch.logical_or(mask, token_to_mask[shuffle1]),
                        token_to_mask[shuffle1].unsqueeze(1)
                    )
                    mask2 = torch.logical_or(
                        torch.logical_or(mask, token_to_mask[shuffle2]),
                        token_to_mask[shuffle2].unsqueeze(1)
                    )
                    
                    projection_1 = model(X, mask1)
                    projection_2 = model(X, mask2)
                else:
                    data_aug_1 = augmenter(X)
                    data_aug_2 = augmenter(X)
                    projection_1 = model(data_aug_1)
                    projection_2 = model(data_aug_2)
                    
                
                train_loss = evaluate_loss(
                    loss_func,
                    [projection_1, projection_2],
                    loss_args
                )

                train_loss.backward()
                optimizer.step()
                train_loss_tot += train_loss.item()
                # verbose print
                if verbose:
                    pbar.set_description(f" train {batch_idx+1:8<}/{len(train_dataloader):8>}")
                    pbar.set_postfix_str(
                        f"train_loss={train_loss_tot/(batch_idx+1):.5f}, "
                        f"val_loss={val_loss_tot:.5f}"
                    )
                    pbar.update()
            train_loss_tot /= batch_idx + 1

            if lr_scheduler != None:
                lr_scheduler.step()

            # Perform validation if validation dataloader were given
            if perform_validation:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for batch_idx, X in enumerate(validation_dataloader):

                        X = X.to(device=device)
                        
                        if both_mask_and_aug:
                            shuffle1 = torch.randperm(token_num)
                            shuffle2 = torch.randperm(token_num)
                            mask1 = torch.logical_or(
                                torch.logical_or(mask, token_to_mask[shuffle1]),
                                token_to_mask[shuffle1].unsqueeze(1)
                            )
                            mask2 = torch.logical_or(
                                torch.logical_or(mask, token_to_mask[shuffle2]),
                                token_to_mask[shuffle2].unsqueeze(1)
                            )
                            
                            data_aug_1 = augmenter(X)
                            data_aug_2 = augmenter(X)
                            
                            projection_1 = model(data_aug_1, mask1)
                            projection_2 = model(data_aug_2, mask2)
                        elif mask_tokens:
            
                            shuffle1 = torch.randperm(token_num)
                            shuffle2 = torch.randperm(token_num)
                            mask1 = torch.logical_or(
                                torch.logical_or(mask, token_to_mask[shuffle1]),
                                token_to_mask[shuffle1].unsqueeze(1)
                            )
                            mask2 = torch.logical_or(
                                torch.logical_or(mask, token_to_mask[shuffle2]),
                                token_to_mask[shuffle2].unsqueeze(1)
                            )
                            
                            projection_1 = model(X, mask1)
                            projection_2 = model(X, mask2)
                        else:
                            data_aug_1 = augmenter(X)
                            data_aug_2 = augmenter(X)
                            projection_1 = model(data_aug_1)
                            projection_2 = model(data_aug_2)
                        
                        val_loss = evaluate_loss(
                            validation_loss_func,
                            [projection_1, projection_2],
                            validation_loss_args,
                        )
                        val_loss_tot += val_loss.item()
                        if verbose:
                            pbar.set_description(
                                f"   val {batch_idx+1:8<}/{len(validation_dataloader):8>}"
                            )
                            pbar.set_postfix_str(
                                f"train_loss={train_loss_tot:.5f}, "
                                f"val_loss={val_loss_tot/(batch_idx+1):.5f}"
                            )
                            pbar.update()

                    val_loss_tot /= batch_idx + 1

        # Deal with earlystopper if given
        if EarlyStopper != None:
            updated_mdl = False
            if EarlyStopper.monitored == "validation":
                curr_monitored = val_loss_tot
            else:
                curr_monitored = train_loss_tot
            EarlyStopper.early_stop(curr_monitored)
            if EarlyStopper.record_best_weights:
                if EarlyStopper.best_loss == curr_monitored:
                    EarlyStopper.rec_best_weights(model)
                    updated_mdl = True
            if EarlyStopper():
                if verbose:
                    print(f"no improvement after {EarlyStopper.patience} epochs.")
                    print(f"Training stopped at epoch {epoch}")
                if EarlyStopper.record_best_weights and not (updated_mdl):
                    EarlyStopper.restore_best_weights(model)
                if return_loss_info:
                    return loss_info
                else:
                    return

        if return_loss_info:
            loss_info[epoch] = [train_loss_tot, val_loss_tot]
    if return_loss_info:
        return loss_info



def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor = None,
    Lambda: float = 5,
    Mu: float = 7.5,
    Nu: float = 1,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    if z2 == None:
        z1, z2 = torch.split(z1, int(z1.shape[0] / 2))

    N, D = z1.shape
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    # invariance loss
    sim_loss = F.mse_loss(z1, z2)

    # variance loss
    std_z1 = torch.sqrt(z1.var(dim=0) + epsilon)
    std_z2 = torch.sqrt(z2.var(dim=0) + epsilon)
    std_loss = (torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))) / 2

    # covariance loss
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z1[range(D), range(D)] = 0.0
    cov_z2 = (z2.T @ z2) / (N - 1)
    cov_z2[range(D), range(D)] = 0.0
    cov_loss = cov_z1.pow_(2).sum() / D + cov_z2.pow_(2).sum() / D
    loss = Lambda * sim_loss + Mu * std_loss + Nu * cov_loss
    return loss