# coding=utf-8

import ml_collections
from configs.default_burgers import get_default_configs

def get_config():

    config = get_default_configs()

    # model
    model = config.model = ml_collections.ConfigDict()
    model.name = 'sf_net'
    model.rank = 5
    model.g_width = 40
    model.g_depth = 2

    # training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 20

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




