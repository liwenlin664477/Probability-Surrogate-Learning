import ml_collections
import torch

def get_default_configs():

    config = ml_collections.ConfigDict()
    config.domain = 'TopOpt'

    # raw data files
    config.datafiles = datafiles = ml_collections.ConfigDict()
    datafiles.path = 'pde_data'
    datafiles.fid_list_tr = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    datafiles.fid_list_te = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    datafiles.ns_list_tr = [512] * 12
    datafiles.ns_list_te = [512] * 12
    datafiles.n_threas = 8
    datafiles.interp = 'cubic'
    datafiles.generate = False

    # dataset
    config.data = data = ml_collections.ConfigDict()
    data.normalize = False
    data.fid_min = 50
    data.fid_max = 80
    data.t_min = 0.0
    data.t_max = 1.0
    data.target_fidelity = 80
    data.fid_list_tr = [50, 60, 70, 80]
    data.ns_list_tr = [256, 128, 64, 32]
    data.fid_list_te = [50, 60, 70, 80]
    data.ns_list_te = [256, 256, 256, 256]
    data.interp = 'bicubic'
    data.fold = 0

    # logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.display = False

    # misc
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return config