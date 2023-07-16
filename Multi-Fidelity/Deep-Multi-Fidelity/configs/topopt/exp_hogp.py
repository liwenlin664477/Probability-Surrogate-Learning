# coding=utf-8

import ml_collections
from configs.default_topopt import get_default_configs

def get_config():

    config = get_default_configs()

    # model
    model = config.model = ml_collections.ConfigDict()
    model.name = 'hogp'
    model.rank = 5

    # training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 5000

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
    optim.scheduler = 'NA'

    return config




