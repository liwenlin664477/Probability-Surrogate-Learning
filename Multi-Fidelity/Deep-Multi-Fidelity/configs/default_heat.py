import ml_collections
import torch

def get_default_configs():

    config = ml_collections.ConfigDict()
    config.domain = 'Heat'

    # raw data files
    config.datafiles = datafiles = ml_collections.ConfigDict()
    datafiles.path = 'pde_data'
    datafiles.fid_list_tr = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160]
    datafiles.fid_list_te = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160]
    datafiles.ns_list_tr = [512] * 20
    datafiles.ns_list_te = [512] * 20
    datafiles.n_threas = 4
    datafiles.interp = 'linear'
    datafiles.generate = False

    # dataset
    config.data = data = ml_collections.ConfigDict()
    data.normalize = True
    data.fid_min = 8
    data.fid_max = 64
    data.t_min = 0.0
    data.t_max = 1.0
    data.target_fidelity = 64
    data.fid_list_tr = [8, 16, 32, 64]
    data.ns_list_tr = [100, 50, 50, 20]
    data.fid_list_te = [8, 16, 32, 64]
    data.ns_list_te = [256,256,256,256]
    data.interp = 'bilinear'
    data.fold = 0

    # logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.display = False

    # misc
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return config