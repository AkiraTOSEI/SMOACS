from src.main_script import main_experiment
import os
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms


main_experiment(
    dir_path = './initial_crystal_candidates/structures_from_MEGNet', # directory of initial crystal structures
    mask_data_path = './mask_arrays/mask_dict_super_common.npz', # atomic mask array
    radii_data_path = None, # ionic radii array
    exp_name = 'S_Cry_random', # experiment name. 
    model_name = 'Crystalformer' , # model name. 'Crystalformer' or 'ALIGNN'
    num_steps=200, # optimization steps
    num_candidate=8, # The number of initial crystal structures
    num_batch_crystal=8, # Batch size to optimize 
    
    # prediction model, loss and criteria for each target values
    prediction_loss_setting_dict ={ 
        'bandgap': {
            'prediction_model':None,
            'loss_function':{
                'func_name':'GapLoss',
                'margin':0.04, # (eV)
                'target_bandgap':2.0, # (eV)
                'mode':'band',
                'coef':1.0, # coefficient for the loss 
            },
            'criteria_max':2.04, # optimized values should be less than this value
            'criteria_min':1.96, # optimizedvalues should be more than this value
        },
        'e_form':{
            'prediction_model':None,
            'loss_function':{
                'func_name':'FormationEnegryLoss',
                'e_form_min':None,
                'e_form_coef':1.0, # coefficient for the loss 
            },
            'criteria_max':-0.5, # optimized values should be less than this value
            'criteria_min':None, # optimizedvalues should be more than this value
        }
    },
    atomic_dictribution_loss_setting_dict = {},
    site_atom_oxidation_dict  = {'Atom_settings': {}, 'use_ionic_radii':False},
    limit_coords_displacement= None, # limiation for displacement of fractional coordindates
    lattice_lr= 0.01, # learning rate for lattice (a,b,c,α,β,γ) optimization
    coords_lr= 0.02, # learning rate for fractional coordinates
    atom_lr= 0.6,  # learning rate for atomic distribution and oxidation state configuration parameters
) 
