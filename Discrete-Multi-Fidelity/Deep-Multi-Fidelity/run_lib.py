
import torch
from torch.optim import Adam
import os
import numpy as np
from tqdm.auto import tqdm

from models.hogp import HoGPR
from models.ifc_ode_2d import IFC2dODE
from models.ifc_gpt_2d import IFC2dGPT
from models.dmf_2d import DMF2d
from models.sf_2d import SFNet2D

from infras.misc import cprint

def run_hogp_2d(
        config,
        dataset,
        logger,
        dirs,
):
    exp_log_workdir, exp_evals_workdir, exp_dicts_workdir = dirs

    Xtr, ytr, ttr = dataset.get_unifid_data_stack(
        selectd_fids=dataset.fid_list_tr,
        train=True,
        scale=config.data.normalize,
    )

    Xte, yte, tte = dataset.get_unifid_data_stack(
        selectd_fids=[dataset.target_fidelity],
        train=False,
        scale=config.data.normalize,
    )

    meshsize = config.data.target_fidelity

    Xtr = Xtr.data.cpu().numpy()
    Xte = Xte.data.cpu().numpy()
    ytr = ytr.data.cpu().numpy().reshape((-1, meshsize, meshsize))
    yte = yte.data.cpu().numpy().reshape((-1, meshsize, meshsize))
    ttr = ttr.data.cpu().numpy()
    tte = tte.data.cpu().numpy()

    Xtrain = np.hstack((Xtr, ttr))
    Xtest = np.hstack((Xte, tte))

    model = HoGPR(Xtrain, ytr, config.model.rank, config.device)

    hist_nrmse_tr, hist_nrmse_te = model.train(
        Xtest, yte,
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
        scheduler=config.optim.scheduler,
        min_lr=config.optim.minimum_learning_rate,
        max_epochs=config.training.epochs,
        test_freq=config.testing.freq,
        logger=logger,
        save_path=exp_dicts_workdir,
    )

    # print(hist_nrmse_tr)
    # print(hist_nrmse_te)

    np.save(os.path.join(exp_evals_workdir, 'nrmse_tr.npy'), hist_nrmse_tr)
    np.save(os.path.join(exp_evals_workdir, 'nrmse_te.npy'), hist_nrmse_te)

    # model = HoGPR(Xtrain, ytr, config.model.rank, config.device)
    #
    # err1, err2, err3, err4, tau = model._callback(
    #     torch.tensor(Xtest), torch.tensor(yte),
    # )
    #
    # cprint('r', '{}-{}-{}-{}'.format(err1, err2, err3, err4, tau))
    #
    # model.load_state(exp_dicts_workdir)
    #
    # err1, err2, err3, err4, tau = model._callback(
    #     torch.tensor(Xtest), torch.tensor(yte),
    # )
    #
    # cprint('g', '{}-{}-{}-{}'.format(err1, err2, err3, err4, tau))


def run_ifc_ode_2d(
        config,
        dataset,
        logger,
        dirs,
):
    exp_log_workdir, exp_evals_workdir, exp_dicts_workdir = dirs

    Xtr_list, ytr_list, ttr_list = dataset.get_data(
        train=True,
        scale=config.data.normalize,
        device=config.device,
    )

    Xte_list, yte_list, tte_list = dataset.get_data(
        train=False,
        scale=config.data.normalize,
        device=config.device,
    )

    inf_fid_model = IFC2dODE(
        in_dim=dataset.input_dim,
        h_dim=config.model.rank,
        s_dim=config.data.fid_max,
        int_steps=config.model.ode_int_steps,
        solver=config.model.ode_solver,
        dataset=dataset,
        g_width=config.model.g_width,
        g_depth=config.model.g_depth,
        f_width=config.model.f_width,
        f_depth=config.model.f_depth,
        A_width=config.model.A_width,
        A_depth=config.model.A_depth,
        interp=config.data.interp
    ).to(config.device)

    optimizer = Adam(inf_fid_model.parameters(), lr=config.optim.learning_rate)

    if config.optim.scheduler == 'CosAnnealingLR':
        cprint('y', 'INFO: Cosine annealing scheduler applied.')
        iterations = config.training.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    elif config.optim.scheduler == 'ReduceLROnPlateau':
        cprint('y', 'INFO: Reduce on plateau scheduler applied.')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=config.optim.minimum_learning_rate)
    elif config.optim.scheduler == 'NA':
        scheduler = None
        cprint('y', 'INFO: no scheduler applied.')
        cprint('y', 'training with no scheduler used...')

    hist_nrmse_tr = []
    hist_nrmse_te = []

    best_rmse = np.inf

    for ie in tqdm(range(config.training.epochs + 1)):

        optimizer.zero_grad()

        loss = inf_fid_model.eval_loss(Xtr_list, ytr_list, ttr_list)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            rmse_list_te = inf_fid_model.eval_rmse(Xte_list, yte_list, tte_list)
            scheduler.step(rmse_list_te[config.data.fid_list_te[-1]])
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        if ie % config.testing.freq == 0:

            with torch.no_grad():
                rmse_list_tr, adjust_rmse = inf_fid_model.eval_rmse(
                    Xtr_list, ytr_list, ttr_list, return_adjust=True)

                rmse_list_te = inf_fid_model.eval_rmse(Xte_list, yte_list, tte_list)


                mae_list_tr = inf_fid_model.eval_mae(Xtr_list, ytr_list, ttr_list)
                mae_list_te = inf_fid_model.eval_mae(Xte_list, yte_list, tte_list)

                # pred_list = inf_fid_model.eval_pred(Xte_list, tte_list)

                nrmse_tr = rmse_list_tr[config.data.fid_list_te[-1]]
                nrmse_te = rmse_list_te[config.data.fid_list_te[-1]]

                nmae_tr = mae_list_tr[config.data.fid_list_te[-1]]
                nmae_te = mae_list_te[config.data.fid_list_te[-1]]

                if nrmse_te < best_rmse:
                    best_rmse = nrmse_te
                    torch.save(inf_fid_model.state_dict(), os.path.join(exp_dicts_workdir, 'model.pt'))

                info_str = '\nepoch={}, nelbo={:.5f}, best_rmse={:.5f}\n'.format(ie, loss.item(), best_rmse)
                info_str += '  - nrmse_tr={}\n'.format(nrmse_tr)
                info_str += '  - nrmse_te={}\n'.format(nrmse_te)
                info_str += '  - nmae_tr={}\n'.format(nmae_tr)
                info_str += '  - nmae_te={}\n'.format(nmae_te)
                logger.info(info_str)

                hist_nrmse_tr.append(nrmse_tr)
                hist_nrmse_te.append(nrmse_te)
            #
        #
    #

    hist_nrmse_tr = np.array(hist_nrmse_tr)
    hist_nrmse_te = np.array(hist_nrmse_te)

    np.save(os.path.join(exp_evals_workdir, 'nrmse_tr.npy'), hist_nrmse_tr)
    np.save(os.path.join(exp_evals_workdir, 'nrmse_te.npy'), hist_nrmse_te)

    # inf_fid_model = IFC2dODE(
    #     in_dim=dataset.input_dim,
    #     h_dim=config.model.rank,
    #     s_dim=config.data.fid_max,
    #     int_steps=config.model.ode_int_steps,
    #     solver=config.model.ode_solver,
    #     dataset=dataset,
    #     g_width=config.model.g_width,
    #     g_depth=config.model.g_depth,
    #     f_width=config.model.f_width,
    #     f_depth=config.model.f_depth,
    #     A_width=config.model.A_width,
    #     A_depth=config.model.A_depth,
    #     interp=config.data.interp
    # ).to(config.device)
    #
    # rmse_list_te = inf_fid_model.eval_rmse(Xte_list, yte_list, tte_list)
    # cprint('r', rmse_list_te)
    #
    # inf_fid_model.load_state_dict(torch.load(os.path.join(exp_dicts_workdir, 'model.pt')))
    #
    #
    # rmse_list_te = inf_fid_model.eval_rmse(Xte_list, yte_list, tte_list)
    # cprint('b', rmse_list_te)

def run_ifc_gpt_2d(
        config,
        dataset,
        logger,
        dirs,
):
    exp_log_workdir, exp_evals_workdir, exp_dicts_workdir = dirs

    Xtr_list, ytr_list, ttr_list = dataset.get_data(
        train=True,
        scale=config.data.normalize,
        device=config.device,
    )

    Xte_list, yte_list, tte_list = dataset.get_data(
        train=False,
        scale=config.data.normalize,
        device=config.device,
    )

    inf_fid_model = IFC2dGPT(
        in_dim=dataset.input_dim,
        h_dim=config.model.rank,
        s_dim=config.data.fid_max,
        int_steps=config.model.ode_int_steps,
        solver=config.model.ode_solver,
        dataset=dataset,
        g_width=config.model.g_width,
        g_depth=config.model.g_depth,
        f_width=config.model.f_width,
        f_depth=config.model.f_depth,
        ode_t=True,
        interp=config.data.interp
    ).to(config.device)

    optimizer = Adam(inf_fid_model.parameters(), lr=config.optim.learning_rate)

    if config.optim.scheduler == 'CosAnnealingLR':
        cprint('y', 'INFO: Cosine annealing scheduler applied.')
        iterations = config.training.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    elif config.optim.scheduler == 'ReduceLROnPlateau':
        cprint('y', 'INFO: Reduce on plateau scheduler applied.')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=config.optim.minimum_learning_rate)
    elif config.optim.scheduler == 'NA':
        scheduler = None
        cprint('y', 'INFO: no scheduler applied.')
        cprint('y', 'training with no scheduler used...')

    hist_nrmse_tr = []
    hist_nrmse_te = []

    best_rmse = np.inf

    for ie in tqdm(range(config.training.epochs + 1)):

        optimizer.zero_grad()

        loss = inf_fid_model.eval_nelbo(Xtr_list, ytr_list, ttr_list)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            rmse_list_te = inf_fid_model.eval_rmse(Xte_list, yte_list, tte_list, ttr_list)
            scheduler.step(rmse_list_te[config.data.fid_list_te[-1]])
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        if ie % config.testing.freq == 0:

            with torch.no_grad():
                rmse_list_tr, adjust_rmse = inf_fid_model.eval_rmse(
                    Xtr_list, ytr_list, ttr_list, ttr_list, return_adjust=True)

                rmse_list_te = inf_fid_model.eval_rmse(Xte_list, yte_list, tte_list, ttr_list)


                mae_list_tr = inf_fid_model.eval_mae(Xtr_list, ytr_list, ttr_list, ttr_list)
                mae_list_te = inf_fid_model.eval_mae(Xte_list, yte_list, tte_list, ttr_list)

                # pred_list = inf_fid_model.eval_pred(Xte_list, tte_list)

                nrmse_tr = rmse_list_tr[config.data.fid_list_te[-1]]
                nrmse_te = rmse_list_te[config.data.fid_list_te[-1]]

                nmae_tr = mae_list_tr[config.data.fid_list_te[-1]]
                nmae_te = mae_list_te[config.data.fid_list_te[-1]]

                if nrmse_te < best_rmse:
                    best_rmse = nrmse_te
                    torch.save(inf_fid_model.state_dict(), os.path.join(exp_dicts_workdir, 'model.pt'))

                info_str = '\nepoch={}, nelbo={:.5f}, best_rmse={:.5f}\n'.format(ie, loss.item(), best_rmse)
                info_str += '  - nrmse_tr={}\n'.format(nrmse_tr)
                info_str += '  - nrmse_te={}\n'.format(nrmse_te)
                info_str += '  - nmae_tr={}\n'.format(nmae_tr)
                info_str += '  - nmae_te={}\n'.format(nmae_te)
                logger.info(info_str)

                hist_nrmse_tr.append(nrmse_tr)
                hist_nrmse_te.append(nrmse_te)
            #
        #
    #

    hist_nrmse_tr = np.array(hist_nrmse_tr)
    hist_nrmse_te = np.array(hist_nrmse_te)

    np.save(os.path.join(exp_evals_workdir, 'nrmse_tr.npy'), hist_nrmse_tr)
    np.save(os.path.join(exp_evals_workdir, 'nrmse_te.npy'), hist_nrmse_te)

    # inf_fid_model = IFC2dGPT(
    #     in_dim=dataset.input_dim,
    #     h_dim=config.model.rank,
    #     s_dim=config.data.fid_max,
    #     int_steps=config.model.ode_int_steps,
    #     solver=config.model.ode_solver,
    #     dataset=dataset,
    #     g_width=config.model.g_width,
    #     g_depth=config.model.g_depth,
    #     f_width=config.model.f_width,
    #     f_depth=config.model.f_depth,
    #     ode_t=True,
    #     interp=config.data.interp
    # ).to(config.device)
    #
    # rmse_list_te = inf_fid_model.eval_rmse(Xte_list, yte_list, tte_list, ttr_list)
    # cprint('r', rmse_list_te)
    #
    # inf_fid_model.load_state_dict(torch.load(os.path.join(exp_dicts_workdir, 'model.pt')))
    #
    #
    # rmse_list_te = inf_fid_model.eval_rmse(Xte_list, yte_list, tte_list, ttr_list)
    # cprint('b', rmse_list_te)

def run_dmf_2d(
        config,
        dataset,
        logger,
        dirs,
):
    exp_log_workdir, exp_evals_workdir, exp_dicts_workdir = dirs

    # Xtr_list, ytr_list, t_list_tr = dataset.get_data(train=True, device=config.device)
    # Xte_list, yte_list, t_list_te = dataset.get_data(train=False, device=config.device)

    Xtr_list, ytr_list, ttr_list = dataset.get_data(
        train=True,
        scale=config.data.normalize,
        device=config.device,
    )

    Xte_list, yte_list, tte_list = dataset.get_data(
        train=False,
        scale=config.data.normalize,
        device=config.device,
    )

    model = DMF2d(
        config,
        nfids=len(list(Xte_list.keys())),
        fids_list=list(Xte_list.keys()),
        in_dim=dataset.input_dim,
        rank=config.model.rank,
        meshes=[config.data.fid_max, config.data.fid_max],
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        activation=config.model.activation,
    ).to(config.device)
    print("datafiles:", config.datafiles.ns_list_te)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data.shape)



    optimizer = Adam(model.parameters(), lr=config.optim.learning_rate, weight_decay=config.optim.weight_decay)

    if config.optim.scheduler == 'CosAnnealingLR':
        cprint('y', 'INFO: Cosine annealing scheduler applied.')
        iterations = config.training.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    elif config.optim.scheduler == 'ReduceLROnPlateau':
        cprint('y', 'INFO: Reduce on plateau scheduler applied.')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=config.optim.minimum_learning_rate)
    elif config.optim.scheduler == 'NA':
        scheduler = None
        cprint('y', 'INFO: no scheduler applied.')
        cprint('y', 'training with no scheduler used...')

    hist_nrmse_tr = []
    hist_nrmse_te = []

    best_rmse = np.inf

    for ie in tqdm(range(config.training.epochs+1)):

        optimizer.zero_grad()

        loss = model.eval_nelbo(Xtr_list, ytr_list, ns=config.training.ns)
        loss.backward()
        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            rmse_list_te = model.eval_rmse(Xte_list, yte_list)
            scheduler.step(rmse_list_te[config.data.fid_list_te[-1]])
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        if ie % config.testing.freq == 0:

            with torch.no_grad():

                fids_rmse_tr = model.eval_rmse(Xtr_list, ytr_list)
                fids_rmse_te = model.eval_rmse(Xte_list, yte_list)
                fids_list = list(Xtr_list.keys())

                # cprint('r', fids_rmse_tr)
                # cprint('r', fids_rmse_te)

                nrmse_tr = fids_rmse_tr[-1].item()
                nrmse_te = fids_rmse_te[-1].item()

                if nrmse_te < best_rmse:
                    best_rmse = nrmse_te
                    torch.save(model.state_dict(), os.path.join(exp_dicts_workdir, 'model.pt'))


                info_str = '\nepoch={}, nelbo={:.5f}, best_rmse={:.5f}\n'.format(ie, loss.item(), best_rmse)
                info_str += '  - nrmse_tr={}\n'.format(nrmse_tr)
                info_str += '  - nrmse_te={}\n'.format(nrmse_te)
                logger.info(info_str)

                hist_nrmse_tr.append(nrmse_tr)
                hist_nrmse_te.append(nrmse_te)
            #
        #
    #

    hist_nrmse_tr = np.array(hist_nrmse_tr)
    hist_nrmse_te = np.array(hist_nrmse_te)

    np.save(os.path.join(exp_evals_workdir, 'nrmse_tr.npy'), hist_nrmse_tr)
    np.save(os.path.join(exp_evals_workdir, 'nrmse_te.npy'), hist_nrmse_te)

    # model = DMF2d(
    #     config,
    #     nfids=len(list(Xte_list.keys())),
    #     fids_list=list(Xte_list.keys()),
    #     in_dim=dataset.input_dim,
    #     rank=config.model.rank,
    #     meshes=[config.data.fid_max, config.data.fid_max],
    #     hidden_dim=config.model.hidden_dim,
    #     hidden_layers=config.model.hidden_layers,
    #     activation=config.model.activation,
    # ).to(config.device)
    #
    # fids_rmse_te = model.eval_rmse(Xte_list, yte_list)
    # cprint('r', fids_rmse_te)
    #
    # model.load_state_dict(torch.load(os.path.join(exp_dicts_workdir, 'model.pt')))
    #
    # fids_rmse_te = model.eval_rmse(Xte_list, yte_list)
    # cprint('b', fids_rmse_te)

def run_sf_2d(
        config,
        dataset,
        logger,
        dirs,
):
    exp_log_workdir, exp_evals_workdir, exp_dicts_workdir = dirs

    Xtr, ytr, _ = dataset.get_unifid_data_stack(
        selectd_fids=dataset.fid_list_tr,
        train=True,
        scale=config.data.normalize,
        device=config.device,
    )

    Xte, yte, _ = dataset.get_unifid_data_stack(
        selectd_fids=[dataset.target_fidelity],
        train=False,
        scale=config.data.normalize,
        device=config.device,
    )

    # cprint('r', Xte.shape)
    # cprint('r', Xte)

    model = SFNet2D(
        in_dim=dataset.input_dim,
        h_dim=config.model.rank,
        s_dim=config.data.fid_max,
        dataset=dataset,
        g_width=config.model.g_width,
        g_depth=config.model.g_depth,
        interp=config.data.interp,
    ).to(config.device)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data.shape)


    optimizer = Adam(model.parameters(), lr=config.optim.learning_rate, weight_decay=config.optim.weight_decay)

    if config.optim.scheduler == 'CosAnnealingLR':
        cprint('y', 'INFO: Cosine annealing scheduler applied.')
        iterations = config.training.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    elif config.optim.scheduler == 'ReduceLROnPlateau':
        cprint('y', 'INFO: Reduce on plateau scheduler applied.')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=config.optim.minimum_learning_rate)
    elif config.optim.scheduler == 'NA':
        scheduler = None
        cprint('y', 'INFO: no scheduler applied.')
        cprint('y', 'training with no scheduler used...')

    hist_nrmse_tr = []
    hist_nrmse_te = []

    best_rmse = np.inf

    for ie in tqdm(range(config.training.epochs+1)):

        optimizer.zero_grad()

        loss = model.eval_loss(Xtr, ytr)
        loss.backward()
        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            rmse_te = model.eval_rmse(Xte, yte)
            scheduler.step(rmse_te)
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        if ie % config.testing.freq == 0:

            with torch.no_grad():

                rmse_tr = model.eval_rmse(Xtr, ytr)
                rmse_te = model.eval_rmse(Xte, yte)

                mae_tr = model.eval_mae(Xtr, ytr)
                mae_te = model.eval_mae(Xte, yte)

                nrmse_tr = rmse_tr.item()
                nrmse_te = rmse_te.item()

                if nrmse_te < best_rmse:
                    best_rmse = nrmse_te
                    torch.save(model.state_dict(), os.path.join(exp_dicts_workdir, 'model.pt'))


                info_str = '\nepoch={}, nelbo={:.5f}, best_rmse={:.5f}\n'.format(ie, loss.item(), best_rmse)
                info_str += '  - nrmse_tr={}\n'.format(nrmse_tr)
                info_str += '  - nrmse_te={}\n'.format(nrmse_te)
                logger.info(info_str)

                hist_nrmse_tr.append(nrmse_tr)
                hist_nrmse_te.append(nrmse_te)
            #
        #
    #

    hist_nrmse_tr = np.array(hist_nrmse_tr)
    hist_nrmse_te = np.array(hist_nrmse_te)

    np.save(os.path.join(exp_evals_workdir, 'nrmse_tr.npy'), hist_nrmse_tr)
    np.save(os.path.join(exp_evals_workdir, 'nrmse_te.npy'), hist_nrmse_te)

    # model = SFNet2D(
    #     in_dim=dataset.input_dim,
    #     h_dim=config.model.rank,
    #     s_dim=config.data.fid_max,
    #     dataset=dataset,
    #     g_width=config.model.g_width,
    #     g_depth=config.model.g_depth,
    #     interp=config.data.interp,
    # ).to(config.device)
    #
    #
    # rmse_te = model.eval_rmse(Xte, yte)
    # cprint('r', rmse_te)
    #
    # model.load_state_dict(torch.load(os.path.join(exp_dicts_workdir, 'model.pt')))
    #
    # rmse_te = model.eval_rmse(Xte, yte)
    # cprint('b', rmse_te)

