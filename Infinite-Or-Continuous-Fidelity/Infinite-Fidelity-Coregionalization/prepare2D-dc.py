import numpy as np


from data.dataset2D import MFData2D

import numpy as np
import copy
import random

from scipy.io import savemat
from infras.misc import *
from infras.utils import *

from data.domain_configs import EXP_DOMAIN_CONFIGS

domain = 'Heat'

dataset = MFData2D(
    domain,
    fid_min     = EXP_DOMAIN_CONFIGS[domain]['fid_min'],
    fid_max     = EXP_DOMAIN_CONFIGS[domain]['fid_max'],
    t_min       = EXP_DOMAIN_CONFIGS[domain]['t_min'],
    t_max       = EXP_DOMAIN_CONFIGS[domain]['t_max'],
    fid_list_tr = EXP_DOMAIN_CONFIGS[domain]['fid_list_tr'],
    fid_list_te = EXP_DOMAIN_CONFIGS[domain]['fid_list_te'],
    ns_list_tr  = EXP_DOMAIN_CONFIGS[domain]['ns_list_tr'],
    ns_list_te  = EXP_DOMAIN_CONFIGS[domain]['ns_list_te'],
)

nfolds = 5
meta_path = 'data_dc'

for fold in range(nfolds):
    
    _, Xtr_list, ytr_list = dataset.get_data_mfhogp(
        fold=fold, train=True, scale=True)
    
    _, Xte_list, yte_list = dataset.get_data_mfhogp(
        fold=fold, train=False, scale=True) 
    
    for Xtr, ytr in zip(Xte_list, yte_list):
        cprint('r', Xtr.shape)
        cprint('r', ytr.shape)
    dc_Xtr = {}
    dc_Xte = {}
    dc_ytr = {}
    dc_yte = {}
    
    nfids = len(Xtr_list)
    print(nfids)
    
    for fid in range(nfids):
        
        Xtr = Xtr_list[fid]
        ytr = ytr_list[fid]
        Xte = Xte_list[fid]
        yte = yte_list[fid]
        
        dc_Xtr[fid] = Xtr
        dc_ytr[fid] = ytr
        dc_Xte[fid] = Xte
        dc_yte[fid] = yte
        
    #


#     data_fold = {
#         'Xtr_list': dc_Xtr,
#         'ytr_list': dc_ytr,
#         'Xte_list': dc_Xte,
#         'yte_list': dc_yte,
#     }

    data_fold = {
        'Xtr_list': Xtr_list,
        'ytr_list': ytr_list,
        'Xte_list': Xte_list,
        'yte_list': yte_list,
    }
    
    data_path = os.path.join(meta_path, domain, 'fold'+str(fold))
    create_path(data_path)
    
    savemat(os.path.join(data_path, 'data.mat'), data_fold)
    
    