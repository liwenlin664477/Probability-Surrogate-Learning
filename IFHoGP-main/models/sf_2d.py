import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.DoubleTensor)

class Net(nn.Module):
    def __init__(self, config, act=nn.Tanh()):

        super(Net, self).__init__()

        buff_layers = []

        for l in range(len(config) - 2):
            in_dim = config[l]
            out_dim = config[l + 1]
            buff_layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            buff_layers.append(nn.Tanh())
        #
        buff_layers.append(nn.Linear(in_features=config[-2], out_features=config[-1]))

        self.net = nn.ModuleList(buff_layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, val=0)
            #
        #

    def forward(self, X):
        h = X
        for layer in self.net:
            h = layer(h)
        #
        return h


class SFNet2D(nn.Module):
    def __init__(self,
                 in_dim,
                 h_dim,
                 s_dim,
                 dataset,
                 g_width,
                 g_depth,
                 interp='bilinear',
                 ):
        super(SFNet2D, self).__init__()

        self.dataset = dataset

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.y_dim = self.dataset.fid_max

        self.interp = interp

        self.register_buffer('dummy', torch.tensor([]))

        g_config = [self.in_dim] + [g_width] * g_depth + [self.h_dim]

        self.g_model = Net(g_config)

        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.log_nu = nn.Parameter(torch.tensor(0.0))

        self.A = nn.Parameter(torch.zeros(self.h_dim + 1, self.s_dim ** 2))
        nn.init.xavier_normal_(self.A[:-1, :])

    def _eval_llh(self, X, y):
        #         h_list = self._batch_forward_h(X_list, t_list)
        #         A_list = self._solve_At(t_list, batch=True)
        h = self.g_model(X)
        Am, Ab = self.A[:-1, :], self.A[-1, :]
        pred = h @ Am + Ab

        N = y.shape[0]

        pred_2d = pred.reshape([N, self.s_dim, self.s_dim])

        interp_pred_2d = torch.nn.functional.interpolate(
            pred_2d.unsqueeze(1),
            size=self.y_dim,
            mode=self.interp,
        ).squeeze(1)

        # print(pred_2d.shape)
        # print(interp_pred_2d.shape)

        interp_pred_1d = interp_pred_2d.reshape([N, -1])

        # print(interp_pred_1d.shape)

        d = y.shape[1]

        llh = 0.5 * d * self.log_tau - \
              0.5 * d * torch.log(2 * torch.tensor(np.pi).to(self.dummy.device)) - \
              0.5 * torch.exp(self.log_tau) * \
              (torch.square(interp_pred_1d - y).sum(1))

        return llh.sum()

    def _eval_prior(self, ):
        param_list = []
        param_list += [torch.flatten(p) for p in self.g_model.parameters()]

        flat_ode_params = torch.cat(param_list)

        dim = flat_ode_params.shape[0]

        lprior = 0.5 * dim * self.log_nu - \
                 0.5 * dim * torch.log(2 * torch.tensor(np.pi)) - \
                 0.5 * torch.exp(self.log_nu) * \
                 (torch.square(flat_ode_params).sum())

        return lprior

    def eval_loss(self, X, y):
        llh = self._eval_llh(X, y)
        lprior = self._eval_prior()
        nlogprob = -(llh + lprior)
        # print(nlogprob)
        return nlogprob

    def predict(self, X):
        with torch.no_grad():
            h = self.g_model(X)
            Am, Ab = self.A[:-1, :], self.A[-1, :]
            pred = h @ Am + Ab

            N = X.shape[0]

            pred_2d = pred.reshape([N, self.s_dim, self.s_dim])

            interp_pred_2d = torch.nn.functional.interpolate(
                pred_2d.unsqueeze(1),
                size=self.y_dim,
                mode=self.interp,
            ).squeeze(1)

            # cprint('r', pred_2d.shape)
            # cprint('r', interp_pred_2d.shape)

            interp_pred_1d = interp_pred_2d.reshape([N, -1])

            return interp_pred_1d

    def eval_pred(self, X):
        pred_1d = self.predict(X)
        N = X.shape[0]
        pred_2d = pred_1d.reshape([N, self.y_dim, self.y_dim])

        return pred_2d.data.cpu().numpy()

    def eval_rmse(self, X, y):
        pred_1d = self.predict(X)

        # print(pred_1d)

        # nrmse_te = torch.sqrt(torch.mean(torch.square(pred_te - Yte))) / torch.sqrt(torch.square(Yte).mean())

        rmse = torch.sqrt(torch.mean(torch.square(y - pred_1d))) / \
               torch.sqrt(torch.square(y).mean())

        return rmse

    def eval_mae(self, X, y):
        pred_1d = self.predict(X)

        mae = torch.abs(y - pred_1d).mean() / torch.abs(y).mean()

        return mae
