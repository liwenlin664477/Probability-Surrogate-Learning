
import torch
import numpy as np
import copy
# import random



# from scipy import interpolate

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

# from data.pde_solvers import *
from data.rawdata import RawDataThreads


# from data.utils import interpolate2d, interpolate3d
# from data.data_configs import data_configs_rawdata, data_configs_interp

#from infras.randutils import *
# from infras.misc import *

from infras.misc import cprint
from infras.utils import interpolate2d, interpolate3d
from infras.randutils import generate_permutation_sequence

# import pickle
from tqdm.auto import tqdm, trange
# from sklearn.model_selection import KFold

class MFData:
    def __init__(self,
                 config,
                 domain,
                 fid_min,
                 fid_max,
                 t_min,
                 t_max,
                 target_fidelity,
                 fid_list_tr=None,
                 fid_list_te=None,
                 ns_list_tr=None,
                 ns_list_te=None,
                ):

        self.config = config
        self.domain = domain
        
        self.rawdata = RawDataThreads(
            config,
            domain=domain,
            fid_list_tr=config.datafiles.fid_list_tr,
            fid_list_te=config.datafiles.fid_list_te,
            ns_list_tr=config.datafiles.ns_list_tr,
            ns_list_te=config.datafiles.ns_list_te,
            preload=config.datafiles.path,
            generate=config.datafiles.generate,
            n_threads=config.datafiles.n_threas,
            thread_idx=None,
        )
        
        self.solver = self.rawdata.solver
        self.input_dim = self.rawdata.input_dim
        self.input_lb = self.rawdata.solver.lb
        self.input_ub = self.rawdata.solver.ub

        self.interp =config.datafiles.interp

        self.target_fidelity = target_fidelity
        self._init_mappings(t_min, t_max, fid_min, fid_max)

        self.fid_list_tr = copy.deepcopy(fid_list_tr)
        self.fid_list_te = copy.deepcopy(fid_list_te)

        self.ns_list_tr = copy.deepcopy(ns_list_tr)
        self.ns_list_te = copy.deepcopy(ns_list_te)

        # cprint('r', self.fid_list_tr)
        # cprint('r', self.fid_list_te)
        #
        # cprint('y', self.ns_list_tr)
        # cprint('y', self.ns_list_te)

        self._validate_data_configs()
        self._extract_init_dataset()


    def _extract_init_dataset(self,):
        
        self.t_list_tr = [self.func_fid_to_t(fid) for fid in self.fid_list_tr]
        self.t_list_te = [self.func_fid_to_t(fid) for fid in self.fid_list_te]

        # cprint('r', self.t_list_tr)
        # cprint('r', self.t_list_te)

        self.dict_fid_to_ns_tr = {}
        self.dict_fid_to_ns_te = {}

        for fid, ns in zip(self.fid_list_tr, self.ns_list_tr):
            self.dict_fid_to_ns_tr[fid] = ns
        for fid, ns in zip(self.fid_list_te, self.ns_list_te):
            self.dict_fid_to_ns_te[fid] = ns

        #self.scaler_X, self.scaler_y = self.rawdata.get_scalers(self.target_fidelity)

        self.dict_fid_to_Xtr = {}
        self.dict_fid_to_ytr = {}

        self.dict_fid_to_Xte = {}
        self.dict_fid_to_yte = {}

        # cprint('r', self.config.data.fold)

        nall = self.rawdata.dict_fid_to_Xtr[self.fid_list_tr[0]].shape[0]
        if self.config.data.fold == 0:
            perm = np.arange(nall)
        else:
            perm = generate_permutation_sequence(N=nall, seed=self.config.data.fold)
        # cprint('c', nall)
        # cprint('c', perm)


        for fid, ns in zip(self.fid_list_tr, self.ns_list_tr):
            Xall = self.rawdata.dict_fid_to_Xtr[fid]
            yall = self.rawdata.dict_fid_to_ytr[fid]
            # Xtr = self.rawdata.dict_fid_to_Xtr[fid][:ns,...]
            # ytr = self.rawdata.dict_fid_to_ytr[fid][:ns,...]
            # Xall = Xall[perm]
            # yall = yall[perm]
            # print(Xall.shape)
            # print(yall.shape)
            Xtr = Xall[perm][:ns, ...]
            ytr = yall[perm][:ns, ...]
            self.dict_fid_to_Xtr[fid] = Xtr
            self.dict_fid_to_ytr[fid] = ytr
        #

        for fid, ns in zip(self.fid_list_te, self.ns_list_te):
            Xte = self.rawdata.dict_fid_to_Xte[fid][:ns,...]
            yte = self.rawdata.dict_fid_to_yte[fid][:ns,...]
            self.dict_fid_to_Xte[fid] = Xte
            self.dict_fid_to_yte[fid] = yte
        #

        # for fid in self.fid_list_tr:
        #     print(self.dict_fid_to_Xtr[fid].shape)
        #     print(self.dict_fid_to_ytr[fid].shape)
        #
        # for fid in self.fid_list_te:
        #     print(self.dict_fid_to_Xte[fid].shape)
        #     print(self.dict_fid_to_yte[fid].shape)
        #

    def _init_mappings(self, t_min, t_max, fid_min, fid_max):
        
        self.fid_min = fid_min
        self.fid_max = fid_max
        self.t_min = t_min
        self.t_max = t_max
        
        self.func_fid_to_t = lambda fid: \
            (fid-fid_min)*(t_max-t_min)/(fid_max-fid_min)
        
        self.func_t_to_fid = lambda t: \
            round((t-t_min)*(fid_max-fid_min)/(t_max-t_min) + fid_min)
        
        self.func_t_to_idx = lambda t: \
            round((t-t_min)*(fid_max-fid_min)/(t_max-t_min))
        
        fid_list_all = [fid for fid in range(self.fid_min, self.fid_max+1)]
        #cprint('r', self.fid_list)
        
        # sanity check 
        t_steps = 1000
        t_span = np.linspace(t_min, t_max, t_steps)
        for i in range(t_span.size):
            t = t_span[i]
            fid = self.func_t_to_fid(t)
            idx = self.func_t_to_idx(t)
            t_rev = self.func_fid_to_t(fid)
            # cprint('r', '{:3f}-{}-{}'.format(t, fid, fid_list_all[idx]))
            err_t = np.abs(t-t_rev)
            # cprint('b', '{:.5f}-{:.5f}-{:.5f}'.format(t, t_rev, err_t))
            if fid != fid_list_all[idx]:
                raise Exception('Check the mappings of fids')
            #
            if err_t >= (t_max-t_min)/(fid_max-fid_min):
                raise Exception('Check the mappings of t')
            #
        #

        cprint('g', 'Sanity check of time-fidelity one-to-one mapping PASSED.')
        
    def _validate_data_configs(self,):
        # validate if require training and testing dataset can be extract from the
        # preload raw dataset
        
        for fid in self.fid_list_tr:
            if fid in list(self.rawdata.dict_fid_to_Xtr.keys()):
                continue
            else:
                raise Exception('Exp training fidelity {} does not exist in the preload dataset'.format(fid))
            #
        #
        
        for fid in self.fid_list_te:
            if fid in list(self.rawdata.dict_fid_to_Xte.keys()):
                continue
            else:
                raise Exception('Exp testing fidelity {} does not exist in the preload dataset'.format(fid))
            #
        #
        
        for fid, nfid in zip(self.fid_list_tr, self.ns_list_tr):
            if nfid > self.rawdata.dict_fid_to_Xtr[fid].shape[0]:
                raise Exception('preload training does not have enough data at fidelity {}'.format(fid))
            #
        #
        
        for fid, nfid in zip(self.fid_list_te, self.ns_list_te):
            if nfid > self.rawdata.dict_fid_to_Xte[fid].shape[0]:
                raise Exception('preload testing does not have enough data at fidelity {}'.format(fid))
            #
        #

        cprint('g', 'Preload data has covered experiment required data VALIDATED.')

    def _normalize_fidelity_data(self, fid, X, y):

        scaler_X, scaler_y = self.rawdata.get_fid_scalers(fid)
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y)

        return X_scaled, y_scaled

    # def _normalize_fidelity_data(self, fid, X, y):
    #
    #     if 'NavierStock' in self.domain:
    #         interp_fn = interpolate3d
    #     else:
    #         interp_fn = interpolate2d
    #
    #     yhf = interp_fn(
    #         y,
    #         fid, self.target_fidelity,
    #         method=self.config.datafiles.interp
    #     )
    #
    #     yhf_scaled = self.scaler_y.transform(yhf)
    #
    #     y_scaled = interp_fn(
    #         yhf_scaled,
    #         self.target_fidelity, fid,
    #         method=self.config.datafiles.interp
    #     )
    #
    #     X_scaled = self.scaler_X.transform(X)
    #
    #     return X_scaled, y_scaled

    def get_data(self, train=True, scale=True, device=torch.device('cpu')):
        
        X_list = {}
        y_list = {}
        
        if train:
            fids_list = self.fid_list_tr
            dict_ns = self.dict_fid_to_ns_tr
            dict_Xs = self.dict_fid_to_Xtr
            dict_ys = self.dict_fid_to_ytr
            copy_t_list = copy.deepcopy(self.t_list_tr)
        else:
            fids_list = self.fid_list_te
            dict_ns = self.dict_fid_to_ns_te
            dict_Xs = self.dict_fid_to_Xte
            dict_ys = self.dict_fid_to_yte
            copy_t_list = copy.deepcopy(self.t_list_te)
        #
        
        t_list = [torch.tensor(ts).to(device) for ts in copy_t_list]

        # print(fids_list)
        # print(dict_ns)
        # print(t_list)

        for fid in fids_list:
            ns = dict_ns[fid]
            Xs = dict_Xs[fid]
            ys = dict_ys[fid].reshape([ns, -1])
            # cprint('w', '--------------------')
            # cprint('w', fid)
            # cprint('w', ns)
            # cprint('w', Xs.shape)
            # cprint('w', ys.shape)
            # cprint('w', '--------------------')

            if scale:
                Xs, ys = self._normalize_fidelity_data(fid, Xs, ys)
            #

            X_list[fid] = torch.tensor(Xs).to(device)
            y_list[fid] = torch.tensor(ys).to(device)

            # cprint('r', Xs.mean(0))
            # cprint('r', ys.mean(0))
            # cprint('b', Xs.std(0))
            # cprint('b', ys.std(0))
            # cprint('g', Xs)
            # cprint('w', ys.shape)
        #

        # cprint('y', t_list)
    
        return X_list, y_list, t_list

    
    # def _unify_fidelity_data(self, fid, X, y, scale=True):
    #
    #     if 'NavierStock' in self.domain:
    #         interp_fn = interpolate3d
    #     else:
    #         interp_fn = interpolate2d
    #
    #     # cprint('r', y.shape)
    #
    #     yhf = interp_fn(
    #         y,
    #         fid, self.target_fidelity,
    #         method=self.config.datafiles.interp
    #     )
    #
    #     # cprint('c', yhf.shape)
    #
    #     if scale:
    #
    #         yhf_scaled = self.scaler_y.transform(yhf)
    #
    #         X_scaled = self.scaler_X.transform(X)
    #
    #         return X_scaled, yhf_scaled
    #     else:
    #         return X, yhf

    def _unify_fidelity_data(self, fid, X, y, scale=True):

        if scale:
            scaler_X, scaler_y = self.rawdata.get_fid_scalers(fid)
            X = scaler_X.transform(X)
            y = scaler_y.transform(y)


        if 'NavierStock' in self.domain:
            interp_fn = interpolate3d
        else:
            interp_fn = interpolate2d

        # cprint('r', y.shape)

        yhf = interp_fn(
            y,
            fid, self.target_fidelity,
            method=self.config.datafiles.interp
        )

        # cprint('c', yhf.shape)

        # if scale:
        #
        #     yhf_scaled = self.scaler_y.transform(yhf)
        #
        #     X_scaled = self.scaler_X.transform(X)
        #
        #     return X_scaled, yhf_scaled
        # else:
        #     return X, yhf

        return X, yhf
        

    def get_unifid_data_stack(self, selectd_fids=[], train=True, scale=True, device=torch.device('cpu')):
        
        X_list = [] 
        y_list = []
        t_list = []
        
        if len(selectd_fids) == 0:
            raise Exception('no selected fidelities given...')
        
        if train:
            fids_list = self.fid_list_tr
            dict_ns = self.dict_fid_to_ns_tr
            dict_Xs = self.dict_fid_to_Xtr
            dict_ys = self.dict_fid_to_ytr
            copy_t_list = copy.deepcopy(self.t_list_tr)
        else:
            fids_list = self.fid_list_te
            dict_ns = self.dict_fid_to_ns_te
            dict_Xs = self.dict_fid_to_Xte
            dict_ys = self.dict_fid_to_yte
            copy_t_list = copy.deepcopy(self.t_list_te)
        #
        
        for fid in selectd_fids:
            # print(fid)
            Xf, yf = self._unify_fidelity_data(fid, dict_Xs[fid], dict_ys[fid], scale)
            X_list.append(torch.tensor(Xf).to(device))
            y_list.append(torch.tensor(yf).to(device))
            tf = torch.tensor(self.func_fid_to_t(fid)).repeat([Xf.shape[0],1]).to(device)
            t_list.append(tf)
        #
        
        Xunif = torch.vstack(X_list)
        yunif = torch.vstack(y_list)
        tunif = torch.vstack(t_list)

        return Xunif, yunif, tunif
    
    # def get_interp_data(self, selectd_fids=[], train=True, scale=True, device=torch.device('cpu')):
    #
    #     X_list = {}
    #     y_list = {}
    #     t_list = []
    #
    #     if len(selectd_fids) == 0:
    #         #raise Exception('no selected fidelities given...')
    #         cprint('y', 'warning, no selected fidelities given..., use default fids list')
    #         if train:
    #             selectd_fids = self.fid_list_tr
    #         else:
    #             selectd_fids = self.fid_list_te
    #         #
    #
    #     if train:
    #         fids_list = self.fid_list_tr
    #         dict_ns = self.dict_fid_to_ns_tr
    #         dict_Xs = self.dict_fid_to_Xtr
    #         dict_ys = self.dict_fid_to_ytr
    #         copy_t_list = copy.deepcopy(self.t_list_tr)
    #     else:
    #         fids_list = self.fid_list_te
    #         dict_ns = self.dict_fid_to_ns_te
    #         dict_Xs = self.dict_fid_to_Xte
    #         dict_ys = self.dict_fid_to_yte
    #         copy_t_list = copy.deepcopy(self.t_list_te)
    #     #
    #
    #     for fid in selectd_fids:
    #         Xf, yf = self._unify_fidelity_data(fid, dict_Xs[fid], dict_ys[fid], scale)
    #
    #         X_list[fid] = torch.tensor(Xf).to(device)
    #         y_list[fid] = torch.tensor(yf).to(device)
    #         t_list.append(torch.tensor(self.func_fid_to_t(fid)).to(device))
    #     #
    #
    #     return X_list, y_list, t_list


    def get_data_mfhogp(self, train, scale):

        X_list, y_list, t_list = self.get_data(train, scale)
        fids_list, fids_X, fids_y = [], [], []

        for fid_t in list(X_list.keys()):
            ns = y_list[fid_t].shape[0]
            # ys = y_list[fid_t].reshape([ns, fid_t, fid_t]).data.cpu().numpy()
            # cprint('r', ys.shape)
            # cprint('r', type(ys))

            ys = y_list[fid_t].data.cpu().numpy()
            # cprint('r', ys.shape)

            lf = fid_t
            hf = self.fid_max

            # x_lf = np.linspace(0, 1, lf)
            # y_lf = np.linspace(0, 1, lf)
            # x_hf = np.linspace(0, 1, hf)
            # y_hf = np.linspace(0, 1, hf)
            #
            # ys_interp = []
            #
            # for n in range(ns):
            #     u = ys[n]
            #     interp_fn = interpolate.interp2d(x_lf, y_lf, u, kind=config.datafiles.interp)
            #     u_hf = interp_fn(x_hf, y_hf)
            #     ys_interp.append(np.expand_dims(u_hf, 0))
            # #
            #
            # ys_interp = np.concatenate(ys_interp)
            # ys_interp_1d = ys_interp.reshape([ns, -1])
            # cprint('b', ys_interp_1d.shape)
            #
            # ys_interp2 = interpolate2d(y_list[fid_t].data.numpy(), lf, hf, config.datafiles.interp)
            # cprint('b', ys_interp2.shape)
            #
            # print(np.sum(np.square(ys_interp_1d-ys_interp2)))

            ys_interp = interpolate2d(ys, lf, hf, self.config.datafiles.interp)
            # cprint('b', ys_interp.shape)


            xs = X_list[fid_t].data.cpu().numpy()
            # print(type(xs))

            fids_list.append(fid_t)
            fids_X.append(xs)
            fids_y.append(ys_interp)
        #

        return fids_list, fids_X, fids_y

        