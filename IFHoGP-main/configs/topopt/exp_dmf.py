# coding=utf-8

import ml_collections
from configs.default_topopt import get_default_configs

def get_config():

    config = get_default_configs()

    # model
    model = config.model = ml_collections.ConfigDict()
    model.name = 'dmf'
    model.rank = 5
    model.hidden_dim = 40
    model.hidden_layers = 2
    model.activation = 'tanh'

    # training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 100
    training.ns = 10

    # testing
    config.testing = testing = ml_collections.ConfigDict()
    testing.freq = 5
    testing.samples = 50

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.learning_rate = 1e-3
    optim.weight_decay = 1e-5
    optim.minimum_learning_rate = 1e-5
    optim.scheduler = 'CosAnnealingLR'

    return config




