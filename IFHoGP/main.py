import torch
import numpy as np
import os
import copy

from absl import app
from absl import flags

from ml_collections.config_flags import config_flags

from infras.misc import cprint, create_path, get_logger
# from data.dataset import MFData
from data.dataset import MFData
from models.hogp import HoGPR

from run_lib import run_hogp_2d, run_ifc_ode_2d, run_ifc_gpt_2d, run_dmf_2d, run_sf_2d

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])

np.random.seed(0)
torch.manual_seed(0)

def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    workdir = os.path.join(
        workdir,
        config.model.name,
        'rank{}'.format(config.model.rank),
        'fold{}'.format(config.data.fold)
    )

    exp_log_workdir = os.path.join(workdir, 'logs')
    exp_evals_workdir = os.path.join(workdir, 'evals')
    exp_dicts_workdir = os.path.join(workdir, 'dicts')
    #
    create_path(exp_log_workdir, verbose=False)
    create_path(exp_evals_workdir, verbose=False)
    create_path(exp_dicts_workdir, verbose=False)

    logger = get_logger(os.path.join(exp_log_workdir, 'logging.txt'), displaying=False)
    logger.info('=============== Experiment Setup ===============')
    logger.info(config)
    logger.info('================================================')

    print("Learning rate:", config.optim.learning_rate)
    print("Weight decay:", config.optim.weight_decay)
    print("Scheduler:", config.optim.scheduler)
    print("Minimum learning rate:", config.optim.minimum_learning_rate)
    print("config.training.epochs:", config.training.epochs)
    print("config.testing.freq:", config.testing.freq)


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

    # print(f"dataset = MFData(\n"
    #       f"    config,\n"
    #       f"    domain={config.domain},\n"
    #       f"    fid_min={config.data.fid_min},\n"
    #       f"    fid_max={config.data.fid_max},\n"
    #       f"    t_min={config.data.t_min},\n"
    #       f"    t_max={config.data.t_max},\n"
    #       f"    target_fidelity={config.data.target_fidelity},\n"
    #       f"    fid_list_tr={config.data.fid_list_tr},\n"
    #       f"    fid_list_te={config.data.fid_list_te},\n"
    #       f"    ns_list_tr={config.data.ns_list_tr},\n"
    #       f"    ns_list_te={config.data.ns_list_te},\n"
    #       f")")

    if config.model.name == 'hogp':
        run_hogp_2d(config, dataset, logger, (exp_log_workdir, exp_evals_workdir, exp_dicts_workdir))
    elif config.model.name == 'ifc_ode':
        run_ifc_ode_2d(config, dataset, logger, (exp_log_workdir, exp_evals_workdir, exp_dicts_workdir))
    elif config.model.name == 'ifc_gpt':
        run_ifc_gpt_2d(config, dataset, logger, (exp_log_workdir, exp_evals_workdir, exp_dicts_workdir))
    elif config.model.name == 'dmf':
        run_dmf_2d(config, dataset, logger, (exp_log_workdir, exp_evals_workdir, exp_dicts_workdir))
    elif config.model.name == 'sf_net':
        run_sf_2d(config, dataset, logger, (exp_log_workdir, exp_evals_workdir, exp_dicts_workdir))
    else:
        raise Exception('Error: no valid model for experiment found.')

if __name__ == '__main__':
    app.run(main)

