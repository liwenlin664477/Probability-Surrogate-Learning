# import os, sys
# import pickle
# import logging
import numpy as np
import torch
from scipy import interpolate


def interpolate2d(y, lf, hf, method):
    assert y.ndim == 2
    N = y.shape[0]

    x1_lf = np.linspace(0, 1, lf)
    x2_lf = np.linspace(0, 1, lf)

    x1_hf = np.linspace(0, 1, hf)
    x2_hf = np.linspace(0, 1, hf)

    y_hf_2d = []

    for i in range(N):
        y2d = y[i, :].reshape([lf, lf])
        fn = interpolate.interp2d(x1_lf, x2_lf, y2d, kind=method)
        y2d_interp = fn(x1_hf, x2_hf)
        # print(y2d_interp.shape)
        y_hf_2d.append(np.expand_dims(y2d_interp, 0))
    #

    y_hf_2d = np.concatenate(y_hf_2d)
    y_hf = y_hf_2d.reshape([N, hf * hf])

    # print(y_hf.shape)

    return y_hf


def interpolate3d(y, lf, hf, method):
    assert y.ndim == 2
    N = y.shape[0]
    d = y.shape[1]
    nt = int(d / (lf * lf))
    # print(N,d,nt)

    x1_lf = np.linspace(0, 1, lf)
    x2_lf = np.linspace(0, 1, lf)

    x1_hf = np.linspace(0, 1, hf)
    x2_hf = np.linspace(0, 1, hf)

    y_hf = []

    for i in range(N):

        ylf_3d = y[i, :].reshape(nt, lf, lf)
        yhf_3d = []

        for t in range(nt):
            ylf_2d = ylf_3d[t, ...]
            # print(ylf_2d.shape)

            fn = interpolate.interp2d(x1_lf, x2_lf, ylf_2d, kind=method)

            yhf_2d = fn(x1_hf, x2_hf)
            yhf_3d.append(np.expand_dims(yhf_2d, 0))
        #

        yhf_3d = np.concatenate(yhf_3d)
        # print(yhf_3d.shape)
        y_hf.append(np.expand_dims(yhf_3d, 0))
    #

    y_hf = np.concatenate(y_hf)
    # print(y_hf.shape)

    y_hf = y_hf.reshape([N, nt * hf * hf])

    return y_hf
