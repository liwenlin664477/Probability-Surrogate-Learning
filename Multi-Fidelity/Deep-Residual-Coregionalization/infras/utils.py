
import os, sys
import pickle
import logging
import numpy as np


def get_logger(logpath, displaying=True, saving=True, debug=False, append=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        if append:
            info_file_handler = logging.FileHandler(logpath, mode="a")
        else:
            info_file_handler = logging.FileHandler(logpath, mode="w+")
        #
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

class PerformMeters(object):
    
    def __init__(self, save_path, logger=None):
    
        self.save_path = save_path
        self.logger = logger
        
        self.cnt_steps = 1
        
        self.active_hist_loss_tr = []
        self.active_hist_loss_te = []
        self.active_hist_rmse_tr = []
        self.active_hist_rmse_te = []
        self.hist_tq = []

        
    def update(self, hist_train, tq):
        
        hist_loss_tr = np.array(hist_train['hist_loss_tr'])
        hist_loss_te = np.array(hist_train['hist_loss_te'])
        hist_rmse_tr = hist_train['hist_rmse_tr']
        hist_rmse_te = hist_train['hist_rmse_te']
        
        self.hist_tq.append(tq)
        
        step_loss_tr = hist_loss_tr[-1]
        step_loss_te = hist_loss_te[-1]
        step_rmse_tr = hist_rmse_tr[-1]
        step_rmse_te = hist_rmse_te[-1]
        
        self.active_hist_loss_tr.append(hist_loss_tr)
        self.active_hist_loss_te.append(hist_loss_te)
        self.active_hist_rmse_tr.append(hist_rmse_tr)
        self.active_hist_rmse_te.append(hist_rmse_te)
        
        if self.logger is not None:    
            self.logger.info('=========================================')
            self.logger.info('             Active Step {} '.format(self.cnt_steps))
            self.logger.info('=========================================')          
            self.logger.info('  # loss_tr={:.6f}'.format(step_loss_tr))
            self.logger.info('  # loss_te={:.6f}'.format(step_loss_te))
            
            self.logger.info('  # Rmse_tr')
            for fid, rmse in step_rmse_tr.items():
                self.logger.info('    - fid={}, rmse={:.6f}'.format(fid, rmse))
                
            self.logger.info('  # Rmse_te')
            for fid, rmse in step_rmse_te.items():
                self.logger.info('    - fid={}, rmse={:.6f}'.format(fid, rmse))
        #
            
        self.cnt_steps += 1
            

    def save(self,):
        
        res = {}
        
        res['active_hist_loss_tr'] = self.active_hist_loss_tr
        res['active_hist_loss_te'] = self.active_hist_loss_te
        res['active_hist_rmse_tr'] = self.active_hist_rmse_tr
        res['active_hist_rmse_te'] = self.active_hist_rmse_te
        res['hist_tq'] = np.array(self.hist_tq)
        
#         res['steps_rmse_tr'] = np.array(self.steps_rmse_tr)
#         res['steps_rmse_te'] = np.array(self.steps_rmse_te)
#         res['steps_nrmse_tr'] = np.array(self.steps_nrmse_tr)
#         res['steps_nrmse_te'] = np.array(self.steps_nrmse_te)
        

        with open(os.path.join(self.save_path, 'error.pickle'), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        
        