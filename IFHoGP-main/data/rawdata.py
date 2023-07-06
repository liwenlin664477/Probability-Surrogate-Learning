import numpy as np
import copy
import random
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm, trange

from data.pde_solvers import *

from infras.misc import cprint
from infras.utils import interpolate2d, interpolate3d


class RawDataThreads:
    def __init__(self,
                 config,
                 domain,
                 fid_list_tr=None,
                 fid_list_te=None,
                 ns_list_tr=None,
                 ns_list_te=None,
                 preload=None,
                 generate=None,
                 n_threads=None,
                 thread_idx=None,
                 seed=0,
            ):

        self.config = config
        self.domain = domain
        
        if domain == 'Poisson':
            self.solver = Poisson()
        elif domain == 'Heat':
            self.solver = Heat()
        elif domain == 'Burgers':
            self.solver = Burgers()
        elif domain == 'TopOpt':
            self.solver = TopOpt()
        elif domain == 'NavierStockPRec':
            self.solver = NavierStockPRec()
        elif domain == 'NavierStockURec':
            self.solver = NavierStockURec()
        elif domain == 'NavierStockVRec':
            self.solver = NavierStockVRec()
        else:
            raise Exception('No PDE solvers found...')
        #
        
        assert self.solver.lb.size == self.solver.ub.size
        
        self.input_dim = self.solver.lb.size
        
        if preload is None:
            raise Exception('Does not specify a path to load or save the data')
        
        data_path = os.path.join(preload, domain)
        create_path(data_path)

        self.n_threads = n_threads
        
        self.fid_list_tr = copy.deepcopy(fid_list_tr)
        self.fid_list_te = copy.deepcopy(fid_list_te)

        self.ns_list_tr = copy.deepcopy(ns_list_tr)
        self.ns_list_te = copy.deepcopy(ns_list_te)

        assert len(self.fid_list_tr) == len(self.ns_list_tr)
        assert len(self.fid_list_te) == len(self.ns_list_te)
        
        if generate is True:
            self.thread_idx = thread_idx
            assert self.thread_idx < self.n_threads
            cprint('y', 'WARNING: generating data thread of {}/{}...'.format(thread_idx+1, n_threads))
            self.gen_data_thread(data_path, seed)
            
        elif generate is False:
            cprint('b', 'Loading saved data threads...')
            sanity = self._sanity_check_datapath(data_path)
            if sanity :
                cprint('g', 'Sanity check of saving paths PASSED, loading the data...')
                self.load_data(data_path)
            else:
                cprint('r', 'Sanity check of saving paths threads FAILED, data NOT loaded')
                raise Exception('Check saved data threads')
            #
        #
       
    def _sanity_check_datapath(self, save_path):

        sanity = True
        
        for fid in self.fid_list_tr:            
            for thread_idx in range(self.n_threads):
                thread_path = os.path.join(save_path, 'train', 'fidelity_'+str(fid), 'part'+str(thread_idx))
                if os.path.isdir(thread_path):
                    cprint('w', '(Train) fid={}, Part{}/{} found at {}'.format(
                        fid, thread_idx+1, self.n_threads, thread_path))
                else:
                    cprint('r', '(Train) fid={}, Part{}/{} missing at {}'.format(
                        fid, thread_idx+1, self.n_threads, thread_path))
                    sanity = False
                #
            #
        #
        
        for fid in self.fid_list_te:            
            for thread_idx in range(self.n_threads):
                thread_path = os.path.join(save_path, 'test', 'fidelity_'+str(fid), 'part'+str(thread_idx))
                if os.path.isdir(thread_path):
                    cprint('w', '(Test) fid={}, Part{}/{} found at {}'.format(
                        fid, thread_idx+1, self.n_threads, thread_path))
                else:
                    cprint('r', '(Test) fid={}, Part{}/{} missing at {}'.format(
                        fid, thread_idx+1, self.n_threads, thread_path))
                    sanity = False
                #
            #
        #
        
        return sanity
            
        
        
    def _get_thread_idx_list(self, fid_list, ns_list):
        
        dict_thread_idx = {}
        
        for fid, ns in zip(fid_list, ns_list):
            ns_thread  = int(ns/self.n_threads)
            idx_lb = ns_thread*self.thread_idx
            idx_ub = ns_thread*self.thread_idx + ns_thread

            if self.thread_idx == self.n_threads-1:
                dict_thread_idx[fid] = [idx_lb, ns]
            else:
                dict_thread_idx[fid] = [idx_lb, idx_ub]
            #
        #
        
        return dict_thread_idx

    def _init_thread_inputs(self, fid_list, ns_list, init_method, seed):       
        dict_inputs = {}
        dict_thread_idx  = self._get_thread_idx_list(fid_list, ns_list)
          
        for ifid, (fid, ns) in enumerate(zip(fid_list, ns_list)):

            Xall = generate_with_bounds(
                N=ns, 
                lb=self.solver.lb, 
                ub=self.solver.ub, 
                method=init_method, 
                #seed=seed+ifid
                seed=seed
            )
            
            idx_lb = dict_thread_idx[fid][0]
            idx_ub = dict_thread_idx[fid][1]

            Xthread = Xall[idx_lb:idx_ub, :]

            dict_inputs[fid] = Xthread
        #
        return dict_inputs
          
    def _get_solns(self, inputs_list, aux_inputs_list=None):
        
        outputs_list = {}
        fids = list(inputs_list.keys())
        
        for fid in tqdm(fids, desc='Gen Solns'):
            Xs = inputs_list[fid]

            if 'NavierStock' in self.domain:
                ys, Xsaux = self.solver.solve(Xs, fid)
                cprint('r', ys.shape)
                cprint('r', Xsaux.shape)
                aux_inputs_list[fid] = Xsaux
            else:
                ys = self.solver.solve(Xs, fid)
                cprint('r', ys.shape)
                
            ns = ys.shape[0]
            
            if ys.ndim == 3:
                ys_flat = ys.reshape([ns,fid*fid])
            elif ys.ndim == 4:
                nframes = ys.shape[1]
                ys_flat = ys.reshape([ns,nframes*fid*fid])
            
            cprint('g', ys_flat.shape)
            outputs_list[fid] = ys_flat
        #
        
        return outputs_list
    
    
    def gen_data_thread(self, save_path, seed):
        
        dict_thread_Xtr = self._init_thread_inputs(
            fid_list = self.fid_list_tr,
            ns_list = self.ns_list_tr,
            init_method = 'sobol',
            seed = seed
        )
        
#         dict_thread_Xtr = self._init_thread_inputs(
#             fid_list = self.fid_list_tr,
#             ns_list = self.ns_list_tr,
#             init_method = 'lhs',
#             seed = seed
#         )
    
        dict_thread_Xte = self._init_thread_inputs(
            fid_list = self.fid_list_te,
            ns_list = self.ns_list_te,
            init_method = 'uniform',
            seed = seed
        )
        
        if 'NavierStock' in self.domain:
            dict_thread_Xaux_tr = {}
            dict_thread_Xaux_te = {}

            didc_thread_ytr = self._get_solns(dict_thread_Xtr, dict_thread_Xaux_tr)
            didc_thread_yte = self._get_solns(dict_thread_Xte, dict_thread_Xaux_te)
        else:
            didc_thread_ytr = self._get_solns(dict_thread_Xtr)
            didc_thread_yte = self._get_solns(dict_thread_Xte)


        for fid in self.fid_list_tr:
            fid_path = os.path.join(save_path, 'train', 'fidelity_'+str(fid))
            thread_path = os.path.join(fid_path, 'part'+str(self.thread_idx))
            create_path(thread_path)
            np.save(os.path.join(thread_path, 'Xs'), dict_thread_Xtr[fid])
            np.save(os.path.join(thread_path, 'ys'), didc_thread_ytr[fid])
            if 'NavierStock' in self.domain:
                np.save(os.path.join(thread_path, 'Xaux'), dict_thread_Xaux_tr[fid])
        #
        
        for fid in self.fid_list_te:
            fid_path = os.path.join(save_path, 'test', 'fidelity_'+str(fid))
            thread_path = os.path.join(fid_path, 'part'+str(self.thread_idx))
            create_path(thread_path)
            np.save(os.path.join(thread_path, 'Xs'), dict_thread_Xte[fid])
            np.save(os.path.join(thread_path, 'ys'), didc_thread_yte[fid])
            if 'NavierStock' in self.domain:
                np.save(os.path.join(thread_path, 'Xaux'), dict_thread_Xaux_tr[fid])
        #
    #
    
    def _load_saved_threads(self, save_path, train=True):
        
        dict_fid_to_X = {}
        dict_fid_to_y = {}
        
        if train:
            fid_list = self.fid_list_tr
            ns_list = self.ns_list_tr
            load_path = os.path.join(save_path, 'train')
        else:
            fid_list = self.fid_list_te
            ns_list = self.ns_list_te
            load_path = os.path.join(save_path, 'test')
        #
        
        for fid, ns in zip(fid_list, ns_list):
            
            Xthread_list = []
            ythread_list = []
            
            for thread_idx in range(self.n_threads):
                thread_path = os.path.join(load_path, 'fidelity_'+str(fid), 'part'+str(thread_idx))
                #cprint('b', thread_path)
                Xthread = np.load(os.path.join(thread_path, 'Xs.npy'))
                ythread = np.load(os.path.join(thread_path, 'ys.npy'))
                Xthread_list.append(Xthread)
                ythread_list.append(ythread)
            #
            
            Xfid = np.concatenate(Xthread_list, axis=0)
            yfid = np.concatenate(ythread_list, axis=0)

            assert Xfid.shape[0] == ns
            assert yfid.shape[0] == ns 
            
            dict_fid_to_X[fid] = Xfid
            dict_fid_to_y[fid] = yfid
        #
        
        
        return dict_fid_to_X, dict_fid_to_y

    def get_fid_scalers(self, fid):

        X = self.dict_fid_to_Xtr[fid]
        y = self.dict_fid_to_ytr[fid]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        scaler_X.fit(X)
        scaler_y.fit(y)

        return scaler_X, scaler_y

    
    # def get_scalers(self, target_fidelity):
    #
    #     cprint('b', 'this holy fucking shit')
    #
    #     Xall = []
    #     yall = []
    #
    #
    #     for fid in self.fid_list_te:
    #         Xte = self.dict_fid_to_Xte[fid]
    #         yte = self.dict_fid_to_yte[fid]
    #         if 'NavierStock' in self.domain:
    #             yte_interp = interpolate3d(
    #                 yte,
    #                 fid, target_fidelity,
    #                 method=self.config.datafiles.interp
    #             )
    #         else:
    #             yte_interp = interpolate2d(
    #                 yte,
    #                 fid, target_fidelity,
    #                 method=self.config.datafiles.interp
    #             )
    #
    #         Xall.append(Xte)
    #         yall.append(yte_interp)
    #
    #
    #     Xall = np.vstack(Xall)
    #     yall = np.vstack(yall)
    #
    #
    #     scaler_X = StandardScaler()
    #     scaler_y = StandardScaler()
    #     scaler_X.fit(Xall)
    #     scaler_y.fit(yall)
    #
    #     return scaler_X, scaler_y
    
    
    def load_data(self, save_path):
        
        self.dict_fid_to_Xtr, self.dict_fid_to_ytr = self._load_saved_threads(save_path, train=True)
        cprint('g', 'Sanity check of loaded training data PASSED')
        self.dict_fid_to_Xte, self.dict_fid_to_yte = self._load_saved_threads(save_path, train=False)
        cprint('g', 'Sanity check of loaded testing data PASSED')
        
    #

