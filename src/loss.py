from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class GapLoss(torch.nn.Module):
    def __init__(
        self, 
        gap_min: Optional[float] = None, 
        gap_max: Optional[float] = None,
        bandgap_loss_coef:float = 1.0,
        mode: str = 'band'
    ):
        super().__init__()
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.mode = mode
        self.coef = bandgap_loss_coef

        if gap_min is None and gap_max is None:
            raise ValueError("gap_min and gap_max cannot be both None")

        if mode not in ['band', 'min', 'max']:
            raise ValueError("mode must be 'band', 'min' or 'max'")
        
        if mode == 'band' and gap_min is not None and gap_max is not None:
            self.mean = (gap_min + gap_max) / 2
            self.margin = (gap_max - gap_min) / 2
        elif mode == 'min' and type(gap_min)==float and gap_max is None:
            pass
        elif mode == 'max' and type(gap_max)==float and gap_min is None:
            pass
        else:
            raise ValueError(f"mode and gap_min, gap_max are not matched! mode: {mode}, gap_min: {gap_min}, gap_max: {gap_max}")
        
    def forward(self, x: torch.Tensor, reduction='none') -> torch.Tensor:
        if self.mode == 'band':
            loss = self.band_loss(x)*self.coef
        elif self.mode == 'min':
            loss = self.min_loss(x)*self.coef
        elif self.mode == 'max':
            loss = self.max_loss(x)*self.coef
        else:
            raise ValueError("mode must be 'band', 'min' or 'max'")

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss
        else:
            raise ValueError("reduction must be 'mean', 'sum' or 'none")


    def min_loss(self,x):
        return torch.clip(x-self.gap_min, min=0)
    
    def max_loss(self, x):
        return  torch.clip(self.gap_max-x, min=0)

    def band_loss(self, x):
        return  torch.clip(torch.abs(x-self.mean)-self.margin, min=0)
    

class FormationEnegryLoss(torch.nn.Module):
    def __init__(
            self,
            e_form_min: Optional[float] = None,
            e_form_loss_coef: float = 1.0,
    ):
        super().__init__()
        self.e_form_min = e_form_min
        self.e_form_coef = e_form_loss_coef

    def forward(self, e_form_pred: torch.Tensor) -> torch.Tensor:
        if self.e_form_min is not None:
            ef_loss = torch.clip(e_form_pred - self.e_form_min, min=0) * self.e_form_coef
        else:
            ef_loss = e_form_pred * self.e_form_coef
        return ef_loss
    

class ToleranceLoss(nn.Module):
    def __init__(self, target_val=0.9):
        super(ToleranceLoss, self).__init__()
        self.target_val = target_val

    def calculate_tolerance_value(
            self,
            num_batch_crystal: int,
            sharpened_ox_mask: torch.Tensor,
            normalized_dist: torch.Tensor,
            radii_tensor: torch.Tensor,
            site_ids: torch.Tensor,
        ):

        # calculate mean radii at each site normalized by the mean of oxides
        radii_per_atom = torch.sum(sharpened_ox_mask.unsqueeze(1) * radii_tensor, dim=2) # shape=(N, num_type_of_atoms)
        site_mean_raddi = torch.sum(normalized_dist * radii_per_atom, dim=1).view(num_batch_crystal, -1)
        # calculate mean raddi at each crystal. A_site: 0, B_site: 1, X_site: 2
        A_site_mask = site_ids.view(num_batch_crystal, -1) == 0
        A_site_raddi = torch.where(A_site_mask, site_mean_raddi , torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device)).sum(dim=1) / A_site_mask.sum(dim=1)
        B_site_mask = site_ids.view(num_batch_crystal, -1) == 1
        B_site_raddi = torch.where(B_site_mask, site_mean_raddi , torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device)).sum(dim=1) / B_site_mask.sum(dim=1)
        X_site_mask = site_ids.view(num_batch_crystal, -1) == 2
        X_site_raddi = torch.where(X_site_mask, site_mean_raddi , torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device)).sum(dim=1) / X_site_mask.sum(dim=1)
        # testing
        torch.testing.assert_close(A_site_mask.sum(dim=1), B_site_mask.sum(dim=1),  atol=1e-5,  rtol=1e-5)
        torch.testing.assert_close(A_site_mask.sum(dim=1)*3, X_site_mask.sum(dim=1),  atol=1e-5,  rtol=1e-5)
        # calculate tolerance and its loss
        tolerance = (A_site_raddi + X_site_raddi) / ((2 ** 0.5) * (B_site_raddi + X_site_raddi))

        return tolerance


    def forward(
            self,
            num_batch_crystal: int,
            sharpened_ox_mask: torch.Tensor,
            normalized_dist: torch.Tensor,
            radii_tensor: torch.Tensor,
            site_ids: torch.Tensor,
        )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute custom loss.

        Args:
            num_batch_crystal (int): The number of crystals in the batch.
            sharpened_ox_mask (torch.Tensor): The sharpened oxide mask tensor.
            normalized_dist (torch.Tensor): The normalized atomic_distribution tensor.
            radii_tensor (torch.Tensor): The inonic radii tensor.
            site_ids (torch.Tensor): The site ids tensor. For perovskite, A_site: 0, B_site: 1, X_site: 2.

        Returns:
            torch.Tensor: Tensor containing the computed loss.
        """
        tolerance_value = self.calculate_tolerance_value(
            num_batch_crystal=num_batch_crystal,
            sharpened_ox_mask=sharpened_ox_mask,
            normalized_dist=normalized_dist,
            radii_tensor=radii_tensor,
            site_ids=site_ids,
        )
        loss = torch.abs(tolerance_value -self.target_val).view(-1,1)
        return loss, tolerance_value


def calculate_loss_from_output(
    pred_dict: Dict[str, torch.Tensor],
    prediction_loss_setting_dict:dict,
    atomic_dictribution_loss_setting_dict:dict,
    num_batch_crystal:int,
    sharpened_ox_mask: torch.Tensor,
    normalized_dist: torch.Tensor,
    site_ids: torch.Tensor,
    radii_tensor: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  
    """
    Calculate the loss from the output of the model.
    """

    total_each_loss = 0.0 #torch.tensor(0.0).to(device)
    loss_value_dict = {}
    for pred_key in pred_dict.keys():
        # loss values per data (shape=(batch_size,))
        loss_key = pred_key.replace('_pred', '')
        key_loss_val = prediction_loss_setting_dict[loss_key]['loss_function']['func'](pred_dict[pred_key])
        loss_value_dict[loss_key+'_loss']  = key_loss_val
        total_each_loss = total_each_loss + key_loss_val.view(-1,1)
        total_each_loss = total_each_loss.view(-1,1)
        

    # tolerance loss
    for loss_key in atomic_dictribution_loss_setting_dict.keys():
        key_loss_val, tipical_value = atomic_dictribution_loss_setting_dict[loss_key]['loss_function']['func'](
            num_batch_crystal,
            sharpened_ox_mask,
            normalized_dist,
            radii_tensor,
            site_ids
        )
        loss_value_dict[loss_key+'_loss'] = key_loss_val
        loss_value_dict[loss_key] = tipical_value
        total_each_loss = total_each_loss + key_loss_val.view(-1,1)
        total_each_loss = total_each_loss.view(-1,1)


    total_loss = torch.nan_to_num(total_each_loss, nan=0.0).mean()
    loss_value_dict['total_loss'] = total_loss
    
    return total_loss, loss_value_dict


def loss_function_initialization(prediction_loss_setting_dict:dict, atomic_dictribution_loss_setting_dict:dict):
    """
    This function initializes the loss function in the prediction_loss_setting_dict.
    """
    for key in prediction_loss_setting_dict.keys():
        if prediction_loss_setting_dict[key]['loss_function']['func_name'] == 'GapLoss':
            target_bandgap = prediction_loss_setting_dict[key]['loss_function']['target_bandgap']
            margin = prediction_loss_setting_dict[key]['loss_function']['margin']
            prediction_loss_setting_dict[key]['loss_function']['func'] = GapLoss(
                gap_min = target_bandgap - margin,
                gap_max = target_bandgap + margin,
                mode = prediction_loss_setting_dict[key]['loss_function']['mode'],
                bandgap_loss_coef = prediction_loss_setting_dict[key]['loss_function']['coef'],
            )
        elif prediction_loss_setting_dict[key]['loss_function']['func_name'] == 'FormationEnegryLoss':
            prediction_loss_setting_dict[key]['loss_function']['func'] = FormationEnegryLoss(
                e_form_min = prediction_loss_setting_dict[key]['loss_function']['e_form_min'],
                e_form_loss_coef = prediction_loss_setting_dict[key]['loss_function']['e_form_coef'],
            )
        else:
            raise NotImplementedError

    for key in atomic_dictribution_loss_setting_dict.keys():
        if atomic_dictribution_loss_setting_dict[key]['loss_function']['func_name'] == 'tolerance':
            tolerance_range = atomic_dictribution_loss_setting_dict[key]['loss_function']['tolerance_range']
            atomic_dictribution_loss_setting_dict[key]['loss_function']['func'] = ToleranceLoss(
               target_val = (tolerance_range[0]+tolerance_range[1])/2
            )
        else:
            raise NotImplementedError

    return prediction_loss_setting_dict, atomic_dictribution_loss_setting_dict
