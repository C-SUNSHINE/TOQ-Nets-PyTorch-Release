#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : action_classification_softmax.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import math
import os
import pickle
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm

from toqnets.datasets.gfootball.visualizer import display_any
from toqnets.datasets.toyota.visualizer import annotate_video
from toqnets.pipeline import start_experiment
from toqnets.utils import average_off_none, plot_confusing_matrix


def in_period(x, y):
    return False if y is None else (x >= y[0] and x <= y[1])


def calc_acc(cor, tot, pct=True):
    if tot == 0:
        return -1.0 if pct else -0.01
    return cor / tot * (100 if pct else 1)


def action_classification_softmax(epoch, model, loader, optimizer, config, hp=None, sp=None, mode=None, device=None,
                                  logger=None, eval=None, save_dir=None, **kwargs):
    epoch_start_time = time.time()
    logger_token_run_epoch = logger.log_begin('run_epoch', "Epoch %d %s" % (epoch, mode))

    model_hp = {
        'mode': mode,
    }

    model_module = model.module if isinstance(model, nn.DataParallel) else model

    # Set-up model hyperparameters

    phase_name = None
    estimate_inequality_parameters = False

    if eval is None:
        phase_name = 'plain'
        if 'normal_period' in hp['train']:
            if in_period(epoch, hp['train']['normal_period']):
                loader.dataset.set_only_regular('train', False)
                loader.dataset.set_only_regular('val', False)
                loader.dataset.set_only_regular('test', False)
                loader.dataset.set_equal_sample('train', False)
                loader.dataset.set_equal_sample('val', False)
                loader.dataset.set_equal_sample('test', False)
                if hasattr(model_module, 'primitives_require_grad'):
                    model_module.primitives_require_grad(True)
                assert phase_name == 'plain'
                phase_name = 'normal'
        if 'pretrain_period' in hp['train']:
            if in_period(epoch, hp['train']['pretrain_period']):
                loader.dataset.set_only_regular('train', True)
                loader.dataset.set_only_regular('val', True)
                loader.dataset.set_only_regular('test', True)
                loader.dataset.set_equal_sample('train', False)
                loader.dataset.set_equal_sample('val', False)
                loader.dataset.set_equal_sample('test', False)
                if hasattr(model_module, 'primitives_require_grad'):
                    model_module.primitives_require_grad(True)
                assert phase_name == 'plain'
                phase_name = 'pretrain'
        if 'finetune_period' in hp['train']:
            if in_period(epoch, hp['train']['finetune_period']):
                if epoch == hp['train']['finetune_period'][0]:
                    sp['lr'] = max(sp['lr'], 0.001)
                loader.dataset.set_only_regular('train', False)
                loader.dataset.set_only_regular('val', False)
                loader.dataset.set_only_regular('test', False)
                loader.dataset.set_equal_sample('train', True)
                loader.dataset.set_equal_sample('val', True)
                loader.dataset.set_equal_sample('test', False)
                if 'finetune_fix_primitive' in hp['train'] and hp['train']['finetune_fix_primitive']:
                    if hasattr(model_module, 'primitives_require_grad'):
                        model_module.primitives_require_grad(False)
                assert phase_name == 'plain'
                phase_name = 'finetune'
        if 'gfootball_finetune_period' in hp['train']:
            if in_period(epoch, hp['train']['gfootball_finetune_period']):
                if epoch == hp['train']['gfootball_finetune_period'][0]:
                    sp['lr'] = max(sp['lr'], 0.001)
                    loader.dataset.set_gfootball_finetune(True)
                    if mode == 'train':
                        estimate_inequality_parameters = True
                loader.dataset.set_equal_sample('train', True)
                loader.dataset.set_equal_sample('val', True)
                loader.dataset.set_equal_sample('test', False)
                # model_module.set_grad('all')
                assert phase_name
                phase_name = 'gfootball_finetune'
    else:
        loader.dataset.set_equal_sample('test', bool(hp['eval']['equal_sample']))
        loader.dataset.set_equal_sample('test', True)

    # print('phase_name =', phase_name)

    if 'beta_decay' in hp['train'] and hp['train']['beta_decay'] is not None:
        beta_epoch = epoch
        if phase_name == 'gfootball_finetune':
            l, r = hp['train']['gfootball_finetune_period']
            beta_epoch = (epoch - l) / (r - l) * 199 + 1
        ds = hp['train']['beta_decay']
        model_hp['beta'] = ((beta_epoch - ds[0][0]) * ds[1][1] + (ds[1][0] - beta_epoch) * ds[0][1]) / (
                ds[1][0] - ds[0][0])
        logger.log_err('run_epoch', "beta = %lf" % model_hp['beta'])

    if 'estimate_inequality_parameters' in hp['train'] and mode == 'train':
        if in_period(epoch, hp['train']['estimate_inequality_parameters']):
            estimate_inequality_parameters = True
    if estimate_inequality_parameters:
        estimating_batches = tqdm(loader)
        estimating_batches.set_description('Estimating parameters')
        model_hp['estimate_parameters'] = True
        for batch_id, data in enumerate(estimating_batches):
            for k in data.keys():
                data[k] = data[k].to(device)
            model(data, hp=model_hp)
        model_module.reset_parameters()
        model_module.zero_grad()
        model_hp.pop('estimate_parameters')
    if 'add_l1loss' in hp['train'] and mode == 'train':
        if in_period(epoch, hp['train']['add_l1loss']):
            model_hp['add_l1loss'] = True

    # if 'binary_logic_layer' in hp['train']:
    #     model_hp['binary_logic_layer'] = in_period(epoch, hp['train']['binary_logic_layer'])

    # Initialize statistic variables and run model

    epoch_loss = 0
    correct = correct_top3 = total = 0
    accumulate_model_time = 0
    batches = tqdm(loader)
    batches.set_description("Epoch %d %s running" % (epoch, mode))

    if mode in ['train', 'val']:
        model.train()
    elif mode in ['test']:
        model.eval()

    if eval is not None:
        y_true_all = []
        y_pred_all = []
        tar_dump = []
        out_dump = []
        sid_dump = []
        result_dump = {}
        n_actions = len(loader.dataset.get_action_list())
        if hasattr(model_module, 'prep_eval'):
            model_module.prep_eval()
        if hasattr(loader.dataset, 'set_clip_sample'):
            loader.dataset.set_clip_sample('test', 'all')
        confusing_matrix = None
        model_hp['eval'] = True
        binary_stat = np.zeros((9, 2))
        if 'RLBench_Fewshot' in config['data']['name']:
            binary_stat = np.zeros(
                (len(loader.dataset.get_original_action_list()), len(loader.dataset.get_action_list())))

    if 'Toyota' in config['data']['name']:
        data_display_count = {}
    else:
        data_display_count = 0

    for batch_id, data in enumerate(batches):
        for k in data.keys():
            data[k] = data[k].to(device)

        if hasattr(loader.dataset, 'formalize_batch'):
            data = loader.dataset.formalize_batch(data)

        batch_model_start_time = time.time()

        outputs = model(data, hp=model_hp)

        out, tar = outputs['output'], outputs['target']
        sid = data['sample_ids']
        model_loss = outputs['loss']

        dim = tar.dim()

        if 'weight' in data:
            weight = data['weight'].float()
        else:
            weight = 1.0

        correct += (torch.eq(out.argmax(dim=dim), tar).float() * weight).sum()
        _, out_top3 = out.topk(min(3, out.size(dim)), dim=-1)
        correct_top3 += (torch.eq(tar.unsqueeze(dim), out_top3).float() * (
            weight.unsqueeze(1) if 'weight' in data else 1.0)).sum()
        total += weight.sum() if 'weight' in data else float(tar.view(-1).size(0))

        if config['hp']['train']['focal_loss']:
            alpha = 1
            gamma = 2
            xent_loss = torch.nn.functional.cross_entropy(out.reshape(-1, out.size(dim)), tar.reshape(-1),
                                                          reduction='none')
            pt = torch.exp(-xent_loss)
            focal_loss = (alpha * (1. - pt) ** gamma * xent_loss)
            losses = focal_loss.view(out.size(0), -1).mean(1)
        else:
            losses = torch.nn.functional.cross_entropy(out.reshape(-1, out.size(dim)), tar.reshape(-1),
                                                       reduction='none').view(out.size(0), -1).mean(1)

        if 'weight' in data:
            losses *= data['weight'].float()

        accumulate_model_time += time.time() - batch_model_start_time

        batch_loss = losses.sum() + model_loss.sum()

        if mode == 'train':
            model.zero_grad()
            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp['clip'])
            optimizer.step()
        else:
            model.zero_grad()
            if optimizer is not None:
                optimizer.zero_grad()
            batch_loss.backward()
            if optimizer is not None:
                optimizer.zero_grad()

        epoch_loss += float(batch_loss)

        if eval is not None:
            if eval > 0:
                if 'Toyota' in config['data']['name']:
                    assert 'index' in data
                    indices = data['index'].detach().cpu().numpy().astype(np.int)
                    ours = out.argmax(dim=dim)
                    display_dir = os.path.join(save_dir, 'visualize', 'eval_samples')
                    action_list = loader.dataset.get_action_list()
                    for i in range(indices.shape[0]):
                        pd_action = action_list[ours[i]]
                        gt_action = action_list[tar[i]]
                        if gt_action not in data_display_count:
                            data_display_count[gt_action] = 0
                        if data_display_count[gt_action] >= eval:
                            continue
                        data_display_count[gt_action] += 1
                        index = int(indices[i])
                        filename = loader.dataset.getitem_from_index(index, 'test', option='filename')
                        annotate_video(filename, os.path.join(display_dir, gt_action))
                        from toqnets.utils import write_text_to_file
                        write_text_to_file("ground_truth: %s, predicted: %s" % (gt_action, pd_action),
                                           os.path.join(display_dir, gt_action, filename + '_result.txt'))

                else:
                    assert data['trajectories'].size(0) >= eval
                    action_list = loader.dataset.get_action_list()
                    display_any(data['trajectories'].cpu()[:eval], 6, 6, data['actions'].cpu()[:eval], action_list,
                                out.cpu()[:eval].argmax(dim=1), os.path.join(save_dir, 'visualize'),
                                out_filter=['shot', 'long_pass'],
                                tar_filter=['shot', 'long_pass'],
                                start_index=data_display_count)
                    data_display_count += eval
            tar_dump.append(tar.cpu().detach().view(-1).numpy().astype(np.int))
            out_dump.append(out.cpu().detach().view(-1, out.size(-1)).numpy().astype(np.float))
            sid_dump.append(sid.cpu().detach().view(-1).numpy().astype(np.int))
            tar_all = tar.cpu().detach().view(-1)
            out_all = out.cpu().detach().view(-1, out.size(-1)).argmax(dim=1)
            eq_all = torch.eq(out_all, tar_all).long()

            # 17 25 correspond
            for i in range(eq_all.size(0)):
                aid = int(data['actions'][i])
                sid = int(data['sample_ids'][i])
                result_dump[(aid, sid)] = eq_all[i] > .5
            # 17 25 correspond

            if confusing_matrix is None:
                confusing_matrix = np.zeros((n_actions, n_actions))
            y_true_all.append(tar_all.numpy().astype(np.int))
            y_pred_all.append(out_all.numpy().astype(np.int))
            few_shot = hasattr(loader.dataset, 'get_new_actions')
            action_list = loader.dataset.get_action_list()
            if few_shot:
                new_actions = loader.dataset.get_new_actions()
            for i in range(tar_all.size(0)):
                if out_all[i] >= n_actions:
                    confusing_matrix[tar_all[i], (tar_all[i] + 1) % n_actions] += 1
                else:
                    confusing_matrix[tar_all[i], out_all[i]] += 1
                if 'original_actions' in data:
                    binary_stat[data['original_actions'][i], out_all[i]] += 1

    desc_string = ''

    if eval is not None:
        # 17 25 correspond
        pickle.dump(result_dump, open(os.path.join(save_dir, 'result_dump.pkl'), 'wb'))
        # 17 25 correspond
        few_shot = hasattr(loader.dataset, 'get_new_actions')
        action_list = loader.dataset.get_action_list()
        new_actions = loader.dataset.get_new_actions() if few_shot else []

        all_cor, all_tot = 0, 0
        all_accs = []
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        for i in range(n_actions):
            all_cor += confusing_matrix[i][i]
            all_tot += confusing_matrix[i].sum()
            all_accs.append(
                float(confusing_matrix[i][i] / confusing_matrix[i].sum()) if confusing_matrix[i].sum() != 0 else None)
        if few_shot:
            reg_cor, reg_tot, new_cor, new_tot = 0, 0, 0, 0
            reg_accs = []
            new_accs = []
            if 'RLBench_Fewshot' not in config['data']['name']:
                for i in range(n_actions):
                    if action_list[i] in new_actions:
                        new_cor += confusing_matrix[i][i]
                        new_tot += confusing_matrix[i].sum()
                        new_accs.append(all_accs[i])
                    else:
                        reg_cor += confusing_matrix[i][i]
                        reg_tot += confusing_matrix[i].sum()
                        reg_accs.append(all_accs[i])
            else:
                original_action_list = loader.dataset.get_original_action_list()
                original_new_action_list = loader.dataset.get_new_actions()
                all_accs = []
                for i in range(binary_stat.shape[0]):
                    action = original_action_list[i]
                    label = loader.dataset.label_action(action)
                    if action in original_new_action_list:
                        new_cor += binary_stat[i, label]
                        new_tot += binary_stat[i].sum()
                        new_accs.append(binary_stat[i, label] / binary_stat[i].sum())
                        all_accs.append(new_accs[-1])
                    else:
                        reg_cor += binary_stat[i, label]
                        reg_tot += binary_stat[i].sum()
                        reg_accs.append(binary_stat[i, label] / binary_stat[i].sum())
                        all_accs.append(reg_accs[-1])
                for i, a in enumerate(original_action_list):
                    print(i, a, loader.dataset.label_action(a))
                print(binary_stat)

            action_ids = [i for i in range(len(action_list))]
            action_reg_ids = [i for i in range(len(action_list)) if action_list[i] not in new_actions]
            action_new_ids = [i for i in range(len(action_list)) if action_list[i] in new_actions]

            acc = {
                'recall': {
                    'reg': {'micro': recall_score(y_true_all, y_pred_all, labels=action_reg_ids, average='micro'),
                            'macro': recall_score(y_true_all, y_pred_all, labels=action_reg_ids, average='macro')},
                    'new': {'micro': recall_score(y_true_all, y_pred_all, labels=action_new_ids, average='micro'),
                            'macro': recall_score(y_true_all, y_pred_all, labels=action_new_ids, average='macro')},
                    'all': {'micro': recall_score(y_true_all, y_pred_all, labels=action_ids, average='micro'),
                            'macro': recall_score(y_true_all, y_pred_all, labels=action_ids, average='macro')},
                },
                'precision': {
                    'reg': {'micro': precision_score(y_true_all, y_pred_all, labels=action_reg_ids, average='micro'),
                            'macro': precision_score(y_true_all, y_pred_all, labels=action_reg_ids, average='macro')},
                    'new': {'micro': precision_score(y_true_all, y_pred_all, labels=action_new_ids, average='micro'),
                            'macro': precision_score(y_true_all, y_pred_all, labels=action_new_ids, average='macro')},
                    'all': {'micro': precision_score(y_true_all, y_pred_all, labels=action_ids, average='micro'),
                            'macro': precision_score(y_true_all, y_pred_all, labels=action_ids, average='macro')},
                },
                'f1': {
                    'reg': {'micro': f1_score(y_true_all, y_pred_all, labels=action_reg_ids, average='micro'),
                            'macro': f1_score(y_true_all, y_pred_all, labels=action_reg_ids, average='macro')},
                    'new': {'micro': f1_score(y_true_all, y_pred_all, labels=action_new_ids, average='micro'),
                            'macro': f1_score(y_true_all, y_pred_all, labels=action_new_ids, average='macro')},
                    'all': {'micro': f1_score(y_true_all, y_pred_all, labels=action_ids, average='micro'),
                            'macro': f1_score(y_true_all, y_pred_all, labels=action_ids, average='macro')},
                },
            }

            for metric in ['recall', 'precision', 'f1']:
                for aset in ['reg', 'new', 'all']:
                    for avg in ['micro', 'macro']:
                        print(metric, aset, avg, acc[metric][aset][avg])
            res_string = '{%s}' % ','.join([
                '%s:{%s}' % (metric, ', '.join([
                    '%s:{%s}' % (aset, ', '.join([
                        '%s:%3.1f' % (avg, acc[metric][aset][avg] * 100.0) for avg in ['micro', 'macro']
                    ])) for aset in ['reg', 'new', 'all']
                ])) for metric in ['recall', 'precision', 'f1']
            ])

            if 'last_run' in kwargs and kwargs['last_run']:
                return average_off_none(reg_accs, -0.01) * 100.

            logger.info(str(all_accs))

            logger.info("Eval acc_per_inst: reg = %.1lf, new = %.1lf, all = %.1lf." % (
                calc_acc(reg_cor, reg_tot), calc_acc(new_cor, new_tot), calc_acc(all_cor, all_tot)))
            logger.info("Eval acc_per_action: reg = %.1lf, new = %.1lf, all = %.1lf." % (
                average_off_none(reg_accs, -0.01) * 100., average_off_none(new_accs, -0.01) * 100.,
                average_off_none(all_accs, -0.01) * 100.))
            logger.info("Eval acc_dict: %s" % repr(acc))

            desc_string = (
                    ("Eval acc_per_inst: reg = %.1lf, new = %.1lf, all = %.1lf." % (
                        calc_acc(reg_cor, reg_tot), calc_acc(new_cor, new_tot), calc_acc(all_cor, all_tot))
                     ) + '\n' +
                    ("Eval acc_per_action: reg = %.1lf, new = %.1lf, all = %.1lf." % (
                        average_off_none(reg_accs, -0.01) * 100., average_off_none(new_accs, -0.01) * 100.,
                        average_off_none(all_accs, -0.01) * 100.)
                     ) + '\n' +
                    res_string
            )
            if 'RLBench_Fewshot' in config['data']['name']:
                if hasattr(loader.dataset, 'get_original_action_list'):
                    oal = loader.dataset.get_original_action_list()
                    for i in range(len(oal)):
                        res_str = oal[i] + ": "
                        for j in range(len(action_list)):
                            res_str += "(%s: %d) " % (action_list[j], binary_stat[i][j])
                        logger.info(res_str)
        else:
            if 'last_run' in kwargs and kwargs['last_run']:
                return average_off_none(all_accs) * 100.
            logger.info("Eval acc_per_inst: all = %.1lf." % (all_cor / all_tot * 100.,))
            logger.info("Eval acc_per_action: all = %.1lf." % (average_off_none(all_accs) * 100.,))

            desc_string = (
                    ("Eval acc_per_inst: all = %.1lf." % (all_cor / all_tot * 100.,)) + '\n' +
                    ("Eval acc_per_action: all = %.1lf." % (average_off_none(all_accs) * 100.,))
            )

        os.makedirs(os.path.join(save_dir, 'visualize'), exist_ok=True)
        plot_confusing_matrix(action_list, confusing_matrix,
                              os.path.join(save_dir, 'visualize', 'confusing_matrix_%d.png' % epoch))
        tar_dump = np.concatenate(tar_dump, axis=0)
        out_dump = np.concatenate(out_dump, axis=0)
        sid_dump = np.concatenate(sid_dump, axis=0)
        np.savez(os.path.join(save_dir, 'visualize', 'eval_debug_seq.npz'), tar=tar_dump, out=out_dump, sid=sid_dump)
        np.savez(os.path.join(save_dir, 'visualize', 'confusing_matrix_raw_%d.npz' % epoch), tar=tar_dump, out=out_dump)

    epoch_size = len(loader.dataset)
    logger.log_end(logger_token_run_epoch)
    return {
        'loss': float(epoch_loss / epoch_size),
        'acc': correct / total,
        'acc_top3': correct_top3 / total,
        'phase': phase_name,
        'desc_string': desc_string
    }


if __name__ == '__main__':
    default_config = {
        'n_epochs': 200,
        'lr': 2e-3,
        'batch_size': 128,
        'cuda': True,
        'seed': 233,
        'save_every': 50,
        'hp': {
            'train': {
                'pretrain_period': (0, 0),  # pre-train period
                'finetune_period': (0, 0),  # fine-tune period
                'normal_period': (0, 0),  # normal period
                'gfootball_finetune_period': (0, 0),
                'add_l1loss': (101, 200),  # period to add l1loss
                'beta_decay': ((1, math.log(1)), (200, math.log(0.001))),
                # beta=log(1) at epoch 1 and log(0.001) at epoch 200, decaying linearly
                'estimate_inequality_parameters': (0, 0),
                # epoch id during which parameters are re-estimated with data, can be set to (1,1) for NLM models
                # 'binary_logic_layer': (0, 0),
                # the period in which binary layers are added to logic machine to round results at each layer
                'finetune_fix_primitive': False,
                # whether to fix parameters of primitive predicates when fine-tune (for NLM models only)
                'focal_loss': False,
            },
            'eval': {
                # 'prerun_model': False,
                # 'show_model': False,
                # 'show_dependency': False,  # {'method': 'weight', 'binary': True},
                # 'use_decision_tree': False,
                'equal_sample': False,
            }
        },
        'trainer': {
            'optimizer': {'name': 'Adam', 'weight_decay': 0.0},
            'clip': 5,
            'lr_decay': {
                # starting at epoch 30, if best model hasn't been updated for more than 5 epochs, reduce lr to lr * 0.9
                'start_epoch': 50,
                'epoch_count': 6,
                'decay': 0.9,
            }
        },
        'evaluator': {
        },
        'loader': {
            'shuffle': True,
            'drop_last': True,
        },
        'data': {
            'name': 'LongVideoNvN',  # name of dataset
        },
        'model': {
            'name': 'NLTL_SAv3',  # name of model
        },
    }
    start_experiment(action_classification_softmax, default_config)
