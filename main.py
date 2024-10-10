from src.main_script import main_experiment
import os
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms


# Crystalformer / perovskite structure
main_experiment(
    exp_name = 'metaric_Silicon',
    model_name = 'Crystalformer' , #'Crystalformer' or 'ALIGNN' or 'Both
    dir_path = './initial_crystal_candidates/one_atom_site_lattice_except_Si',
    mask_data_path = './mask_arrays/ionic_mask_dict_super_common.npz',
    radii_data_path = './mask_arrays/ionic_radii_dict_super_common.npz',

    num_steps=200,
    num_candidate=len(os.listdir('./initial_crystal_candidates/one_atom_site_lattice_except_Si')),
    num_batch_crystal=len(os.listdir('./initial_crystal_candidates/one_atom_site_lattice_except_Si')),
    prediction_loss_setting_dict ={
        'bandgap': {
            'prediction_model':None,
            'loss_function':{
                'func_name':'GapLoss',
                'margin':0.00, # (eV)
                'target_bandgap':0.0, # (eV)
                'mode':'band',
                'coef':1.0,
            },
            'criteria_max':0.0, # optimized values should be less than this value
            'criteria_min':0.0, # optimizedvalues should be more than this value
        },
        'e_form':{
            'prediction_model':None,
            'loss_function':{
                'func_name':'FormationEnegryLoss',
                'e_form_min':None,
                'e_form_coef':1.0,
            },
            'criteria_max':-0.5, # optimized values should be less than this value
            'criteria_min':None, # optimizedvalues should be more than this value
        }
    },
    atomic_dictribution_loss_setting_dict = {},
    site_atom_oxidation_dict  = {
            'Atom_settings': {
                '*': {'element':'Si', 'site_id':-999} # '*' is wildcard. It means all elements are converted to Silicon.
            }, 
            'use_ionic_radii':False,
    },
    atom_lr = 0.0,
    lattice_lr = 0.01,
    coords_lr = 0.01,
    neutral_check=False,
)    

df = pd.read_csv('./results/metaric_Silicon/result.csv').sort_values(['loss_bandgap_onehot', 'e_form_onehot']).head(30)
d = np.load('./results/metaric_Silicon/result.npz', allow_pickle=True)
for lattice_id in df.index.values:
        fname = f"Optimized_based_on_{d['original_fnames'][lattice_id]}"
        atom = Atoms.from_poscar(os.path.join('./results/metaric_Silicon/poscar', fname))
        print(f'initial lattice_id:{d["original_fnames"][lattice_id]}  ', atom.lattice.abc, atom.lattice.angles, atom.coords,"  pred ef(eV):",df.loc[lattice_id,"e_form_onehot"])


# Crystalformer / random structure
main_experiment(
    dir_path = './initial_crystal_candidates/structures_from_MEGNet',
    mask_data_path = './mask_arrays/mask_dict_super_common.npz',
    radii_data_path = None,
    exp_name = 'S_Cry_random_bg',
    model_name = 'Crystalformer' ,
    num_steps=200,
    num_candidate=8,
    num_batch_crystal=8, 
    
    prediction_loss_setting_dict ={
        'bandgap': {
            'prediction_model':None,
            'loss_function':{
                'func_name':'GapLoss',
                'margin':0.04, # (eV)
                'target_bandgap':2.0, # (eV)
                'mode':'band',
                'coef':1.0,
            },
            'criteria_max' : 2.04,
            'criteria_min' : 1.96,
        }
    },
    atomic_dictribution_loss_setting_dict = {},
    site_atom_oxidation_dict  = {'Atom_settings': {}, 'use_ionic_radii':False},
    limit_coords_displacement= None, # 0.15 for perovskite, None for non-perovskite
    atom_lr= 6.0,
    lattice_lr= 0.01,
    coords_lr= 0.02,
)    


# Crystalformer / random structure
main_experiment(
    dir_path = './initial_crystal_candidates/structures_from_MEGNet',
    mask_data_path = './mask_arrays/mask_dict_super_common.npz',
    radii_data_path = None,
    exp_name = 'S_Cry_random_ef',
    model_name = 'Crystalformer' ,
    num_steps=200,
    num_candidate=8,
    num_batch_crystal=8, 
    
    prediction_loss_setting_dict ={
        'e_form':{
            'alignn_model_name':"mp_e_form_alignnn",
            'prediction_model':None,
            'loss_function':{
                'func_name':'FormationEnegryLoss',
                'e_form_min':None,
                'e_form_coef':1.0,
            },
            'criteria_max': -0.5, # optimized values should be less than this value
            'criteria_min': None, # optimizedvalues should be more than this value
        },

    },
    atomic_dictribution_loss_setting_dict = {},
    site_atom_oxidation_dict  = {'Atom_settings': {}, 'use_ionic_radii':False},
    ALIGNN_num_update_graphs = None, #calculate_update_steps(num_steps=200, num_update=4),# np.arange(1,200),
    limit_coords_displacement= None, # 0.15 for perovskite, None for non-perovskite
    atom_lr= 6.0,
    lattice_lr= 0.01,
    coords_lr= 0.02,
)    

# Crystalformer / perovskite structure
main_experiment(
    exp_name = 'S_Cry_perov_ELM_limited',
    model_name = 'Crystalformer' , #'Crystalformer' or 'ALIGNN' or 'Both
    dir_path = './initial_crystal_candidates/random_perovskite_2',
    mask_data_path = './mask_arrays/ionic_mask_dict_super_common.npz',
    radii_data_path = './mask_arrays/ionic_radii_dict_super_common.npz',

    num_steps=200,
    num_candidate=len(os.listdir('./initial_crystal_candidates/random_perovskite_2')), # the number of initial structures
    num_batch_crystal=len(os.listdir('./initial_crystal_candidates/random_perovskite_2'))//2, # batch size
    prediction_loss_setting_dict ={
        'bandgap': {
            'alignn_model_name':"mp_gappbe_alignnn",
            'prediction_model':None,
            'loss_function':{
                'func_name':'GapLoss',
                'margin':0.04, # (eV)
                'target_bandgap':2.0, # (eV)
                'mode':'band',
                'coef':1.0,
            },
            'criteria_max': 2.04,
            'criteria_min': 1.96,
        },
        'e_form':{
            'alignn_model_name':"mp_e_form_alignnn",
            'prediction_model':None,
            'loss_function':{
                'func_name':'FormationEnegryLoss',
                'e_form_min':None,
                'e_form_coef':1.0,
            },
            'criteria_max': -0.5, # optimized values should be less than this value
            'criteria_min': None, # optimizedvalues should be more than this value
        }
    },
    atomic_dictribution_loss_setting_dict = {
        'tolerance':{
            'loss_function':{
                'func_name':'tolerance',
                'coef':1.0,
                'tolerance_range':(0.8, 1.0),
            },
            'criteria_max':1.0, # optimized values should be less than this value
            'criteria_min':0.8, # optimizedvalues should be more than this value
        },
    },
    site_atom_oxidation_dict  = {
            'Atom_settings': {
                'Ba': {'ox_patterns':[2], 'site_id':0}, 
                'Ti': {'element':['Mn', 'Ti'], 'ox_patterns':[4],'site_id':1}, 
                'O': {'ox_patterns':[-2], 'site_id':2},
            },
            'use_ionic_radii':True
    },
    ALIGNN_num_update_graphs = None, #calculate_update_steps(num_steps=200, num_update=4),# np.arange(1,200),
    limit_coords_displacement= 0.15, # 0.15 for perovskite, None for non-perovskite
    angle_range = (85, 95),
    perovskite_evaluation = True,
    neutral_check=False,
    atom_lr= 6.0,
    lattice_lr= 0.01,
    coords_lr= 0.02,
)   
