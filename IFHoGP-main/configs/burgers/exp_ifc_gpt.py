# coding=utf-8

import ml_collections
from configs.default_burgers import get_default_configs

def get_config():

    config = get_default_configs()

    # model
    model = config.model = ml_collections.ConfigDict()
    model.name = 'ifc_gpt'
    model.rank = 5
    model.ode_int_steps = 2
    model.ode_solver = 'dopri5'
    model.g_width = 40
    model.g_depth = 2
    model.f_width = 40
    model.f_depth = 2
    model.kernel = 'RBF'

    # training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 7

    # testing
    config.testing = testing = ml_collections.ConfigDict()
    testing.freq = 5
    testing.samples = 50

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.learning_rate = 1e-2
    optim.weight_decay = 1e-5
    optim.minimum_learning_rate = 1e-3
    optim.scheduler = 'ReduceLROnPlateau'

    return config




