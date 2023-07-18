import numpy as np
import os
import subprocess
import sys
# import argparse
# import copy
from subprocess import CalledProcessError

import fire


from infras.configs import *
from infras.misc import *

class ConfigBaseline(Config):
    
    domain = None
    rank   = None
    method = None
    fold   = None
    
    def __init__(self,):
        super(ConfigBaseline, self).__init__()
        self.config_name = 'SFGP'
        
# def run_command(command):
#     process = subprocess.Popen(command, stdout=subprocess.PIPE)
#     while True:
#         output = process.stdout.readline()
#         if output == '***terminate***' and process.poll() is not None:
#             break
#         if output:
#             print(output.strip().decode("utf-8"))
#         #
#     #
#     rc = process.poll()
#     return rc

def run_command(cmd):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')

    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)

def evaluate(**kwargs):
    
    #kwargs={'domain':'Heat', 'rank':5, 'method':'kpca', 'fold':1}

    exp_config = ConfigBaseline()
    exp_config.parse(kwargs)
    
    if exp_config.method == 'dc':
        data_path = os.path.join('exp_data_dc', exp_config.domain, 'fold'+str(exp_config.fold))
        #data_path = os.path.join('DRC-pde-data', exp_config.domain, 'fold' + str(exp_config.fold))
    else:
        data_path = os.path.join('exp_data', exp_config.domain, 'fold'+str(exp_config.fold))

    res_path = os.path.join(
        '__res_{}__'.format(exp_config.domain.lower()),
        exp_config.method.lower(),
        'rank'+str(exp_config.rank), 
        'fold'+str(exp_config.fold)
    )
    create_path(res_path)
    print(res_path)
    
    data_name = os.path.join(data_path, 'data.mat')
    res_name = os.path.join(res_path, 'results.mat')

    matlab_cmd = 'train_'+exp_config.method+'('+str(exp_config.rank) + ', ' +\
                    '\'' + data_name + '\'' ', ' +\
                    '\'' + res_name + '\'' + '); quit force;'

    command = ["matlab", "-nodesktop", "-r", matlab_cmd]

    run_command(command)


    
if __name__=='__main__':
    
    fire.Fire(evaluate)
