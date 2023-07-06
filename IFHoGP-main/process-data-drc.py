import torch
import numpy as np
import os
import copy

from absl import app
from absl import flags

from ml_collections.config_flags import config_flags

from infras.misc import cprint, create_path, get_logger
from data.dataset import MFData
from infras.utils import interpolate2d

from scipy import interpolate
from scipy.io import savemat

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
# flags.mark_flags_as_required(["workdir", "config"])
flags.mark_flags_as_required(["config"])

np.random.seed(0)
torch.manual_seed(0)


def main(argv):
    config = FLAGS.config

    dataset = MFData(
        config,
        domain=config.domain,
        fid_min=config.data.fid_min,
        fid_max=config.data.fid_max,
        t_min=config.data.t_min,
        t_max=config.data.t_max,
        target_fidelity=config.data.target_fidelity,
        fid_list_tr=config.data.fid_list_tr,
        fid_list_te=config.data.fid_list_te,
        ns_list_tr=config.data.ns_list_tr,
        ns_list_te=config.data.ns_list_te,
    )

    _, Xtr_list, ytr_list = dataset.get_data_mfhogp(train=True, scale=config.data.normalize)
    _, Xte_list, yte_list = dataset.get_data_mfhogp(train=False, scale=config.data.normalize)



    nfids = len(Xtr_list)

    buff_Xtr = np.empty((nfids), dtype=np.object_)
    buff_ytr = np.empty((nfids), dtype=np.object_)
    buff_Xte = np.empty((nfids), dtype=np.object_)
    buff_yte = np.empty((nfids), dtype=np.object_)

    for i in range(nfids):
        buff_Xtr[i] = Xtr_list[i]
        buff_ytr[i] = ytr_list[i]
        buff_Xte[i] = Xte_list[i]
        buff_yte[i] = yte_list[i]

    data_fold = {
        'Xtr_list': buff_Xtr,
        'ytr_list': buff_ytr,
        'Xte_list': buff_Xte,
        'yte_list': buff_yte,
    }

    save_path = os.path.join('DRC-pde-data', config.domain, 'fold{}'.format(config.data.fold))
    create_path(save_path)
    savemat(os.path.join(save_path, 'data.mat'), data_fold)



if __name__ == '__main__':
    app.run(main)
