#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pipeline.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import argparse
import gc
import json
import os
import pickle
import shutil
import time
from copy import deepcopy

import numpy as np
import torch
from jaclearn.mldash import MLDashClient
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import toqnets.datasets as datasets
import toqnets.models as models
from toqnets.config_update import ConfigUpdate, update_config, get_config_update
from toqnets.datasets.gfootball.visualizer import display
from toqnets.datasets.utils import DatasetModeWrapper
from toqnets.logging import get_logger, set_output_file
from toqnets.nn.utils import num_trainable_params
from toqnets.utils import plot_confusing_matrix

logger = get_logger(__file__)


def make_dataset(data_config):
    dataset = getattr(datasets, data_config['name'])
    dataset_obj = dataset(data_config)
    train_dataset = DatasetModeWrapper(dataset_obj, 'train')
    val_dataset = DatasetModeWrapper(dataset_obj, 'val')
    test_dataset = DatasetModeWrapper(dataset_obj, 'test')
    return train_dataset, val_dataset, test_dataset


def complete_config(config, config_update, new_config=True):
    config_update = {k: config_update[k] for k in config_update.keys()}
    if 'data' not in config_update:
        config_update['data'] = ConfigUpdate()
    new_data_name = config_update['data']['name'] if 'name' in config_update['data'] else None
    if new_config or new_data_name is not None:
        data_name = new_data_name if new_data_name is not None else config['data']['name']
        config['data'] = getattr(datasets, data_name).complete_config(config_update['data'])
    else:
        config['data'] = getattr(datasets, config['data']['name']).complete_config(config_update['data'],
                                                                                   default_config=config['data'])
    config_update.pop('data')

    if 'model' not in config_update:
        config_update['model'] = ConfigUpdate()
    new_model_name = config_update['model']['name'] if 'name' in config_update['model'] else None
    if new_config or new_model_name is not None:
        model_name = new_model_name if new_model_name is not None else config['model']['name']
        config['model'] = getattr(models, model_name).complete_config(config_update['model'])
    else:
        config['model'] = getattr(models, config['model']['name']).complete_config(config_update['model'],
                                                                                   default_config=config['model'])
    config_update.pop('model')

    update_config(config, config_update)


def main(processor, default_config, config_update=None, args=None, options=None,
         save_dir=os.path.join('dumps', 'temp')):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if config_update is None:
        config_update = {}
    if options is None:
        options = {}
    os.makedirs(save_dir, exist_ok=True)

    if 'new' in options:
        assert 'eval' not in options, "--eval can not be used with --new or --cont"
        if 'force' not in options:
            should_clear = input("Are you sure to clear all the data under %s? yes/no: " % save_dir).upper() in ["YES",
                                                                                                                 "Y"]
        else:
            should_clear = True
        if should_clear:
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                logger.log_err('main', "option new: Removing %s from %s" % (filename, save_dir))
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        else:
            exit()

    set_output_file(os.path.join(save_dir, 'log.txt' if 'eval' not in options else 'eval_log.txt'))

    ########################################
    #             Make  Config             #
    ########################################

    log_token_make_config = logger.log_begin('main', 'Make config')
    if 'eval' in options:
        config_filename = os.path.join(save_dir, 'config.json')
        config_file = open(config_filename)
        logger.log_err('main', "Loading from default stored config %s." % config_filename)
        config = json.load(config_file)
    else:
        if 'config' in options:
            logger.log_err('main', "Loading from stored config %s" % options['config'])
            config = json.load(open(options['config'], 'r'))
        else:
            config = default_config
    complete_config(config, config_update, new_config='config' not in options and 'eval' not in options)
    logger.log_end(log_token_make_config)

    if 'eval' not in options:
        json.dump(config, open(os.path.join(save_dir, 'config.json'), 'w'), indent=2)

    config_str = json.dumps(config, indent=2)

    logger.info("Config is %s" % config_str)

    if 'run_name' in options:
        run_name = options['run_name']
    else:
        run_name = time.strftime('%Y-%m-%d-%H-%M-%S')

    if 'eval' not in options:
        run_name = 'train-%s-%s' % (run_name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    else:
        run_name = 'eval-%s-%s' % (run_name, time.strftime('%Y-%m-%d-%H-%M-%S'))

    logger.critical('Initializing MLDash.')
    mldash = MLDashClient('dumps')
    if not options['debug']:
        mldash.init(
            desc_name=config['data']['name'],
            expr_name=config['model']['name'],
            run_name=run_name,
            args=args,
            configs=config
        )
        pass
        # mldash.update(
        #     metainfo_file=os.path.join(save_dir, 'config.json'),
        #     log_file=os.path.join(save_dir, 'log.txt' if 'eval' not in options else 'eval_log.txt'),
        # )

    ########################################
    #              Setup Cuda              #
    ########################################

    if not torch.cuda.is_available():
        if config['cuda']:
            logger.log_err('main', "No available cuda!")
        config['cuda'] = False
    if config['cuda']:
        torch.cuda.manual_seed(config['seed'])
    device = torch.device("cuda:0" if config['cuda'] else "cpu")

    ########################################
    #             Build  Model             #
    ########################################

    log_token_build_model = logger.log_begin('main', 'Build model')

    net = getattr(models, config['model']['name'])
    model = net(config['model'])
    logger.log_err('__main__', '#trainable parameters is ' + str(num_trainable_params(model)))
    # input()
    if config['cuda']:
        model = nn.DataParallel(model)
    model = model.to(device)

    logger.log_end(log_token_build_model)

    ########################################
    #             Load Dataset             #
    ########################################

    log_token_make_dataset = logger.log_begin('main', 'Load Dataset')

    train_dataset, val_dataset, test_dataset = make_dataset(config['data'])
    loader_shuffle = config['loader']['shuffle']
    loader_kwargs = {
        'drop_last': config['loader']['drop_last'],
        'batch_size': config['batch_size']
    }
    if config['cuda']:
        loader_kwargs.update({'num_workers': 1, 'pin_memory': True})
    loader = {
        'train': DataLoader(train_dataset, shuffle=loader_shuffle, **loader_kwargs),
        'val': DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        'test': DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    }

    logger.log_end(log_token_make_dataset)

    # loader['train'].dataset.test()
    # exit()

    ########################################
    #           Train/Run  Model           #
    ########################################

    n_epochs = config['n_epochs']

    if 'cont' not in options:
        start_epoch = 1
        end_epoch = n_epochs
        sp = {
            'lr': config['lr'],
            'best_loss': None,
            'epoch_since_best': 0,
        }
    else:
        from_epoch = int(options['cont'])
        model_state_filename = os.path.join(save_dir, 'states', config['model']['name'] + '_%d.pth' % from_epoch)
        logger.log_err('main', "option cont: Loading from epoch %d: %s" % (from_epoch, model_state_filename))
        model.load_state_dict(torch.load(model_state_filename))
        sp_filename = os.path.join(save_dir, 'states', config['model']['name'] + '_%d.pkl' % from_epoch)
        with open(sp_filename, 'rb') as sp_fin:
            sp = pickle.load(sp_fin)
        start_epoch = from_epoch + 1
        end_epoch = n_epochs

    if 'eval' in options:
        from_epoch = options['eval']
        from_epoch = 0 if from_epoch == 'best' else int(from_epoch)
        if from_epoch == 0:
            logger.log_err('main', "option cont: Loading best model")
            model_state_filename = os.path.join(save_dir, 'states', config['model']['name'] + '_best.pth')
        else:
            logger.log_err('main', "option cont: Loading from epoch %d" % from_epoch)
            model_state_filename = os.path.join(save_dir, 'states', config['model']['name'] + '_%d.pth' % from_epoch)
        model.load_state_dict(torch.load(model_state_filename))
        start_epoch = end_epoch = from_epoch
        epoch = from_epoch

    sp_print_list = ['lr']

    if 'check_model_capacity' in options and options['check_model_capacity']:
        n_parameters = int(num_trainable_params(model))
        input("Total number of parameters is %d, press enter to continue" % (n_parameters,))

    logger.log_err('main', "Start running models")

    if 'eval' not in options:
        if args.fast_train:
            train_modes = ['train', 'val']
        else:
            train_modes = ['train', 'val', 'test']

        trainer_config = config['trainer']
        phase_name = None
        for epoch in range(start_epoch, end_epoch + 1):
            # Set up optimizer
            optimizer_name = trainer_config['optimizer']['name']
            assert optimizer_name in ('Adam', 'SGD')
            optimizer_args = deepcopy(trainer_config['optimizer'])
            optimizer_args.pop('name')
            optimizer = getattr(torch.optim, optimizer_name)(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=sp['lr'], **optimizer_args
            )
            # Set up hp
            hp = deepcopy(config['hp'])
            hp.update({
                'clip': trainer_config['clip']
            })
            sp_str = "  |" + '|'.join(["%s: %s" % (x, str(sp[x])) for x in sp_print_list])

            # Run epoch
            logger.log_err('main', "Epoch %d: %s" % (epoch, sp_str))
            epoch_result = {}
            is_new_phase = False

            for mode in train_modes:
                result = processor(
                    epoch, model, loader[mode], optimizer, config,
                    hp=hp, sp=sp, mode=mode, device=device, logger=logger, eval=None, save_dir=save_dir
                )

                # TODO make this output format to be returned by processor
                logger.info("Epoch %d: %s_loss = %f, %s_acc = %f %s_acc_top3 = %f" % (
                    epoch, mode, result['loss'], mode, result['acc'], mode, result['acc_top3']))
                epoch_result[mode] = result
                if 'phase' in result:
                    if phase_name != result['phase']:
                        phase_name = result['phase']
                        is_new_phase = True

                if not options['debug']:
                    pass  # mldash.log_metric('epoch', epoch, desc=False, expr=False)
                    for key, value in result.items():
                        if key.startswith('loss'):
                            pass  # mldash.log_metric_min(mode + '/' + key, float(value), desc=False)
                    for key, value in result.items():
                        if key.startswith('acc'):
                            pass  # mldash.log_metric_max(mode + '/' + key, float(value), desc=False)

            # Finish epoch
            if is_new_phase:
                sp['best_loss'] = None
            is_best_model = False
            if sp['best_loss'] is None or epoch_result['val']['loss'] < sp['best_loss']:
                sp['best_loss'] = epoch_result['val']['loss']
                is_best_model = True
                sp['epoch_since_best'] = 0
            else:
                sp['epoch_since_best'] += 1
                if trainer_config['lr_decay'] is not None:
                    if epoch >= trainer_config['lr_decay']['start_epoch'] and sp['epoch_since_best'] >= \
                            trainer_config['lr_decay']['epoch_count']:
                        sp['epoch_since_best'] = 0
                        sp['lr'] *= trainer_config['lr_decay']['decay']

            # Save model
            if is_best_model:
                os.makedirs(os.path.join(save_dir, 'states'), exist_ok=True)
                model_state_filename = os.path.join(save_dir, 'states', config['model']['name'] + '_best.pth')
                torch.save(model.state_dict(), model_state_filename)
            if epoch % config['save_every'] == 0 or epoch == end_epoch:
                os.makedirs(os.path.join(save_dir, 'states'), exist_ok=True)
                model_state_filename = os.path.join(save_dir, 'states', config['model']['name'] + '_%d.pth' % epoch)
                torch.save(model.state_dict(), model_state_filename)
                sp_filename = os.path.join(save_dir, 'states', config['model']['name'] + '_%d.pkl' % epoch)
                with open(sp_filename, 'wb') as sp_fin:
                    pickle.dump(sp, sp_fin)
            gc.collect()

    if 'eval' not in options:
        set_output_file(os.path.join(save_dir, 'eval_log.txt'))

    hp = config['hp']
    logger.log_err('main', "Evaluate epoch %d:" % (epoch,))
    epoch_result = {}

    for mode in ['test']:
        result = processor(
            epoch, model, loader[mode], None, config,
            hp=hp, sp=None, mode=mode, device=device, logger=logger, eval=options['eval_samples'], save_dir=save_dir
        )
        logger.info("Epoch %d: %s_loss = %f, %s_acc = %f %s_acc_top3 = %f" % (
            epoch, mode, result['loss'], mode, result['acc'], mode, result['acc_top3']))
        epoch_result[mode] = result

        if 'desc_string' in result:
            pass
            # mldash.update(run_description=result['desc_string'])

        for key, value in result.items():
            if key.startswith('loss'):
                pass
                # mldash.log_metric_min(mode + '/' + key, float(value), desc=False)
        for key, value in result.items():
            if key.startswith('acc'):
                pass
                # mldash.log_metric_max(mode + '/' + key, float(value), desc=False)


def start_experiment(processor, default_config):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trial', type=str, default=None,
                        help="trial of experiment, save dir will be at dumps/<trial>/")
    parser.add_argument('--run_name', type=str, default=None, help="run name for ML Dash")
    parser.add_argument('--config', type=str, default=None,
                        help="If specified with a file, will load config from that file.")
    parser.add_argument('--toy', action='store_true', default=False,
                        help="If on, will use small dataset for debugging.")
    parser.add_argument('--fast-train', action='store_true', default=False, help="Fast Train.")

    parser.add_argument('--eval', type=str, default=None, help="eval a specific epoch, 0 for best model.")
    parser.add_argument('--eval_samples', type=int, default=0, help="eval a specific number of samples.")

    parser.add_argument('--new', action='store_true', default=False,
                        help="If on, will start new training session for that trial.")
    parser.add_argument('--cont', type=str, default=None,
                        help="If set on with a epoch number, will continue to train that trial and that epoch.")
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help="If on, won't ask when modifying anything.")

    parser.add_argument('--debug', action='store_true', default=False,
                        help="If on, will focus on pipeline debuging, e.g. won't generate MLDash records.")
    parser.add_argument('--last-run', action='store_true', default=False,
                        help="Special case.")
    parser.add_argument('--check_model_capacity', action='store_true', default=False,
                        help="If on, show capacity and ask for continue before training.")

    args, argv = parser.parse_known_args()
    config_update = ConfigUpdate()
    get_config_update(argv, config_update)

    options = {}
    if args.cont is not None:
        options['cont'] = args.cont
    if args.new:
        options['new'] = True
    if args.config is not None:
        options['config'] = args.config
    if args.eval is not None:
        options['eval'] = args.eval
    options['eval_samples'] = args.eval_samples
    if args.run_name is not None:
        options['run_name'] = args.run_name
    options['force'] = args.force
    options['debug'] = args.debug
    if args.check_model_capacity:
        options['check_model_capacity'] = True

    if 'run_name' in options:
        run_name = options['run_name']
    else:
        run_name = time.strftime('%Y-%m-%d-%H-%M-%S')
        if 'eval' not in options:
            run_name = 'train-%s' % run_name
        else:
            run_name = 'eval-%s' % run_name
    options['run_name'] = run_name

    if args.trial is None:
        args.trial = run_name

    main(processor, default_config, config_update, args=args, options=options,
         save_dir=os.path.join('dumps', args.trial))


def default_processor(epoch, model, loader, optimizer, config, hp=None, sp=None, mode=None, device=None, logger=None,
                      eval=None,
                      save_dir=None):
    epoch_start_time = time.time()
    logger_token_run_epoch = logger.log_begin('run_epoch', "Epoch %d %s" % (epoch, mode))
    epoch_loss = 0
    correct = correct_top3 = total = 0
    accumulate_model_time = 0
    batches = tqdm(loader)
    batches.set_description("Epoch %d %s running" % (epoch, mode))

    criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    if eval is not None:
        confusing_matrix = None

    for batch_id, data in enumerate(batches):
        for k in data.keys():
            data[k] = data[k].to(device)

        batch_model_start_time = time.time()

        out, tar = model(data, hp=hp)

        dim = tar.dim()

        correct += (torch.eq(out.argmax(dim=dim), tar).float()).sum()
        _, out_top3 = out.topk(3, dim=-1)
        correct_top3 += (torch.eq(tar.unsqueeze(dim), out_top3).float()).sum()
        total += float(tar.view(-1).size(0))
        losses = criterion(out.reshape(-1, out.size(dim)), tar.reshape(-1)).view(out.size(0), -1).mean(1)

        accumulate_model_time += time.time() - batch_model_start_time

        batch_loss = losses.sum()

        if mode == 'train':
            model.zero_grad()
            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp['clip'])
            optimizer.step()
        else:
            batch_loss.backward()
        epoch_loss += float(batch_loss)

        if eval is not None:
            if batch_id == 0 and eval > 0:
                assert data['trajectories'].size(0) >= eval
                display(data['trajectories'].cpu()[:eval], data['actions'].cpu()[:eval],
                        out.cpu()[:eval].argmax(dim=dim),
                        os.path.join(save_dir, 'visualize'))
            n_actions = out.size(dim)
            out_all = out.cpu().view(-1, n_actions).argmax(dim=1)
            tar_all = tar.cpu().view(-1)

            if confusing_matrix is None:
                confusing_matrix = np.zeros((n_actions, n_actions))
            for i in range(tar_all.size(0)):
                confusing_matrix[tar_all[i], out_all[i]] += 1

    if eval is not None:
        for i in range(n_actions):
            confusing_matrix[i] /= confusing_matrix[i].sum()
        action_list = loader.dataset.config['actions']
        plot_confusing_matrix(action_list, confusing_matrix,
                              os.path.join(save_dir, 'visualize', 'confusing_matrix.png'))

    epoch_size = len(loader.dataset)
    logger.log_end(logger_token_run_epoch)
    return {
        'loss': float(epoch_loss / epoch_size),
        'acc': correct / total,
        'acc_top3': correct_top3 / total
    }


if __name__ == '__main__':
    default_config = {
        'n_epochs': 200,
        'lr': 1e-4,
        'batch_size': 512,
        'cuda': True,
        'seed': 233,
        'save_every': 10,
        'hp': {
        },
        'trainer': {
            'optimizer': {'name': 'adam'},
            'clip': 10,
            'lr_decay': {
                'start_epoch': 30,
                'epoch_count': 5,
                'decay': 0.75,
            }
        },
        'evaluator': {
        },
        'loader': {
            'shuffle': True,
            'drop_last': True,
        },
        'data': {
            'name': 'TrajectorySingleActionNvN',
        },
        'model': {
            'name': 'GraphTemporalConv_SA',
        },
    }
    start_experiment(default_processor, default_config)
