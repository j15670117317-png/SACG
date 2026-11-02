# version 1, context sample rate is same as long term. now chage it to short-term sample rate
'''
Divide future into short-horizon(anticipation) and long-horizon(future)
--config_file configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml
--test 1 --config_file configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml MODEL.CHECKPOINT checkpoints/THUMOS/cmert_long256_work4_kinetics_1x/epoch-9.pth MODEL.LSTR.INFERENCE_MODE batch
'''
import sys
sys.path.append('./src')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import torch.utils.data as data
import os.path as osp
from bisect import bisect_right
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict, defaultdict
import math
import copy
import os.path as osp
import argparse
import json

from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.utils.ema import build_ema
from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.optimizers import build_optimizer
from rekognition_online_action_detection.optimizers import build_scheduler
from rekognition_online_action_detection.evaluation.postprocessing import postprocessing as default_pp
from rekognition_online_action_detection.criterions import build_criterion
from sklearn.metrics import average_precision_score
from rekognition_online_action_detection.utils.registry import Registry

from rekognition_online_action_detection.datasets import build_data_loader, build_dataset
# for models
from rekognition_online_action_detection.models import transformer as tr
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.evaluation import compute_result_new, compute_result
try:
    from external.rulstm.RULSTM.utils import (get_marginal_indexes, marginalize, softmax,
                                                        topk_accuracy_multiple_timesteps,
                                                        topk_recall_multiple_timesteps,
                                                        tta)
except:
    raise ModuleNotFoundError

def do_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          ema,
                          device,
                          checkpointer,
                          logger):

    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    s1_w = cfg.SOLVER.get('S1_W', 0.2)
    s2_w = cfg.SOLVER.get('S2_W', 1.0)
    f_w = cfg.SOLVER.get('F_W', 0.5)

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        losses_dict = {}
        for l_name in ['tot', 'ant_cls', 'det_cls', 'fut_cls']:
            losses_dict[l_name] = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        pred_scores, ant_pred_scores, fut_pred_scores = [], [], []
        gt_targets, ant_gt_targets, fut_gt_targets  = [], [], []

        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)
            if not training:
                ema.apply_shadow()

            with torch.set_grad_enabled(training):
                pbar = tqdm(data_loaders[phase],
                            desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                for batch_idx, data in enumerate(pbar, start=1):
                    batch_size = data[0].shape[0]
                    det_target, ant_target, fut_target = data[-1]

                    loss_names = list(zip(*cfg.MODEL.CRITERIONS))[0][0]
                    scores, fut_scores = model(*[x.to(device) for x in data[:-1]])
                    tot_target = det_target.to(device)
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        tot_target = torch.cat((det_target, ant_target), dim=1).to(device)

                    for i, detant_score in enumerate(scores):
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            det_score, ant_score = detant_score[:, :cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES, :], \
                                                   detant_score[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                            det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            ant_score = ant_score.reshape(-1, cfg.DATA.NUM_CLASSES)

                        detant_score = detant_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        if i == 0:
                            detant_loss = s1_w * criterion[loss_names](detant_score, tot_target.reshape(-1, cfg.DATA.NUM_CLASSES))
                            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                                det_loss = s1_w * criterion['MCE'](det_score, det_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                                ant_loss = s1_w * criterion['MCE'](ant_score, ant_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                        else:
                            detant_loss += s2_w * criterion[loss_names](detant_score, tot_target.reshape(-1, cfg.DATA.NUM_CLASSES))
                            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                                det_loss += s2_w * criterion['MCE'](det_score,
                                                              det_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                                ant_loss += s2_w * criterion['MCE'](ant_score,
                                                              ant_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                    fut_loss = torch.FloatTensor([0]).to(scores[0].device)[0]
                    for i, fut_score in enumerate(fut_scores):
                        fut_score = fut_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        if i == 0:
                            fut_loss = f_w * criterion['MCE'](fut_score, fut_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                        else:
                            fut_loss += f_w * criterion['MCE'](fut_score, fut_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))

                    loss = detant_loss + fut_loss
                    losses_dict['tot'][phase] += loss.item() * batch_size
                    losses_dict['fut_cls'][phase] += fut_loss.item() * batch_size
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        losses_dict['ant_cls'][phase] += ant_loss.item() * batch_size
                        losses_dict['det_cls'][phase] += det_loss.item() * batch_size

                    if training:
                        optimizer.zero_grad()
                        if loss.item() != 0:
                            loss.backward()
                            optimizer.step()
                            ema.update()
                            scheduler.step()
                    else:
                        # Prepare for evaluation
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            det_score = det_score.softmax(dim=1).cpu().tolist()
                            det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            pred_scores.extend(det_score)
                            gt_targets.extend(det_target)

                            ant_score = ant_score.softmax(dim=1).cpu().tolist()
                            ant_target = ant_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            ant_pred_scores.extend(ant_score)
                            ant_gt_targets.extend(ant_target)
                        else:
                            det_score = detant_score.softmax(dim=1).cpu().tolist()
                            det_target = tot_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            pred_scores.extend(det_score)
                            gt_targets.extend(det_target)

                        if len(fut_scores) > 0 :
                            fut_score = fut_score.softmax(dim=1).cpu().tolist()
                            fut_target = fut_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            fut_pred_scores.extend(fut_score)
                            fut_gt_targets.extend(fut_target)
        end = time.time()

        # Output log for current epoch
        log = []
        log.append('Epoch {:2}'.format(epoch))
        train_log = '[train loss]'
        for k, v in losses_dict.items():
            train_log += ' {}: {:.3f},'.format(k, v['train'] / len(data_loaders['train'].dataset))
        log.append(train_log)

        if 'test' in cfg.SOLVER.PHASES:
            # Compute result
            det_result = compute_result['perframe'](cfg, gt_targets, pred_scores, )
            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                ant_result = compute_result['perframe'](cfg, ant_gt_targets, ant_pred_scores, )
            if len(fut_pred_scores)> 0:
                fut_result = compute_result['perframe'](cfg, fut_gt_targets, fut_pred_scores, )
            test_log = '[test loss]'
            for k, v in losses_dict.items():
                test_log += ' {}: {:.3f},'.format(k, v['test'] / len(data_loaders['test'].dataset))
            log.append(test_log)
            log.append('[mAP] det: {:.3f}， ant: {:.3f}, fut: {:.3f} '.format(det_result['mean_AP'],
                               ant_result['mean_AP'] if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0 else -1,
                                                    fut_result['mean_AP'] if len(fut_pred_scores) >0 else -1))

        log.append('running time: {:.2f} sec'.format(end - start, ))
        logger.info(' | '.join(log))

        # Save checkpoint for model and optimizer
        if epoch % cfg.SOLVER.SAVE_EVERY == 0 and epoch >= 6:
            checkpointer.save(epoch, model, optimizer)

        if not training:
            ema.restore()

        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()
def do_ek100_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          ema,
                          device,
                          checkpointer,
                          logger):
    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    s1_w = cfg.SOLVER.get('S1_W', 0.2)
    s2_w = cfg.SOLVER.get('S2_W', 1.0)
    f_w = cfg.SOLVER.get('F_W', 0.1)
    n_w = cfg.SOLVER.get('N_W', 1.0)
    v_w = cfg.SOLVER.get('V_W', 1.0)
    marginal_w = cfg.SOLVER.get('MG_W', 0.1)
    ant_w = cfg.SOLVER.get('ANT_W', 0)

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        losses_dict = {}
        for l_name in ['tot', 'ant_cls', 'det_cls', 'fut_cls']:
            losses_dict[l_name] = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        # det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        # fut_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []
        pred_scores_by_session = defaultdict(list)
        gt_targets_by_session = defaultdict(list)
        pred_verb_scores_by_session = defaultdict(list)
        gt_verb_targets_by_session = defaultdict(list)
        pred_noun_scores_by_session = defaultdict(list)
        gt_noun_targets_by_session = defaultdict(list)
        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)
            if not training:
                ema.apply_shadow()
                pred_scores_by_session.clear()
                gt_targets_by_session.clear()
                pred_verb_scores_by_session.clear()
                gt_verb_targets_by_session.clear()
                pred_noun_scores_by_session.clear()
                gt_noun_targets_by_session.clear()
            with torch.set_grad_enabled(training):
                pbar = tqdm(data_loaders[phase], desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                for batch_idx, data in enumerate(pbar, start=1):
                    batch_size = data[0].shape[0]
                    loss_names = list(zip(*cfg.MODEL.CRITERIONS))[0]

                    if cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES <= 0:
                        det_target, noun_target, verb_target = data[-1][0].to(device), data[-1][1].to(device), \
                            data[-1][2].to(device)
                        det_target_test = det_target[:, :cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES, :]
                        ant_target = det_target[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                        ant_noun_target = noun_target[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                        ant_verb_target = verb_target[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]

                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        ant_target = ant_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        ant_noun_target = ant_noun_target.reshape(-1, ant_noun_target.shape[-1])
                        ant_verb_target = ant_verb_target.reshape(-1, ant_verb_target.shape[-1])
                        verb_target = verb_target.reshape(-1, verb_target.shape[-1])
                        noun_target = noun_target.reshape(-1, noun_target.shape[-1])

                        det_score, noun_score, verb_score = model(*[x.to(device) for x in data[:-1]])
                        det_score_test = det_score[:, :cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES, :]
                        ant_score = det_score[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                        det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        ant_score = ant_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_loss = criterion['MCE_EQL'](det_score, det_target)
                        det_loss += ant_w * criterion['MCE_EQL'](ant_score, ant_target)

                        ant_verb_score = verb_score[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                        ant_noun_score = noun_score[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                        ant_verb_score = ant_verb_score.reshape(-1, ant_verb_score.shape[-1])
                        ant_noun_score = ant_noun_score.reshape(-1, ant_noun_score.shape[-1])
                        verb_score = verb_score.reshape(-1, verb_score.shape[-1])
                        noun_score = noun_score.reshape(-1, noun_score.shape[-1])

                        verb_loss = criterion['MCE'](verb_score, verb_target)
                        verb_loss += ant_w * criterion['MCE'](ant_verb_score, ant_verb_target)
                        noun_loss = criterion['MCE'](noun_score, noun_target)
                        noun_loss += ant_w * criterion['MCE'](ant_noun_score, ant_noun_target)

                        if 'MARGINAL' in criterion:
                            verb_loss += marginal_w * criterion['MARGINAL'](det_score, verb_target, 'verb')
                            noun_loss += marginal_w * criterion['MARGINAL'](det_score, noun_target, 'noun')

                        det_loss += v_w * verb_loss + n_w * noun_loss
                        losses_dict['det_cls'][phase] += det_loss.item() * batch_size

                    else:
                        det_target, noun_target, verb_target, \
                            fut_target, fut_noun_target, fut_verb_target = data[-1][0].to(device), data[-1][1].to(
                            device), \
                            data[-1][2].to(device), \
                            data[-1][3].to(device), data[-1][4].to(device), \
                            data[-1][5].to(device)
                        det_target_test = det_target[:, :cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES, :]
                        ant_target = det_target[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                        ant_noun_target = noun_target[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                        ant_verb_target = verb_target[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]

                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        ant_target = ant_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        ant_noun_target = ant_noun_target.reshape(-1, ant_noun_target.shape[-1])
                        ant_verb_target = ant_verb_target.reshape(-1, ant_verb_target.shape[-1])
                        verb_target = verb_target.reshape(-1, verb_target.shape[-1])
                        noun_target = noun_target.reshape(-1, noun_target.shape[-1])

                        fut_target = fut_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        fut_noun_target = fut_noun_target.reshape(-1, fut_noun_target.shape[-1])
                        fut_verb_target = fut_verb_target.reshape(-1, fut_verb_target.shape[-1])

                        det_scores, fut_scores, noun_score, fut_noun_score, verb_score, fut_verb_score = model(
                            *[x.to(device) for x in data[:-1]])
                        for i, det_score in enumerate(det_scores):
                            det_score_test = det_score[:, :cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES, :]
                            ant_score = det_score[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                            det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            ant_score = ant_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            if i == 0:
                                det_loss = s1_w * criterion['MCE_EQL'](det_score, det_target)
                                det_loss += s1_w * ant_w * criterion['MCE_EQL'](ant_score, ant_target)
                            else:
                                det_loss += s2_w * criterion['MCE_EQL'](det_score, det_target)
                                det_loss += s2_w * ant_w * criterion['MCE_EQL'](ant_score, ant_target)

                        ant_verb_score = verb_score[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                        ant_noun_score = noun_score[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                        ant_verb_score = ant_verb_score.reshape(-1, ant_verb_score.shape[-1])
                        ant_noun_score = ant_noun_score.reshape(-1, ant_noun_score.shape[-1])
                        det_verb_score = verb_score.reshape(-1, verb_score.shape[-1])
                        det_noun_score = noun_score.reshape(-1, noun_score.shape[-1])

                        verb_loss = criterion['MCE'](det_verb_score, verb_target)
                        verb_loss += ant_w * criterion['MCE'](ant_verb_score, ant_verb_target)
                        noun_loss = criterion['MCE'](det_noun_score, noun_target)
                        noun_loss += ant_w * criterion['MCE'](ant_noun_score, ant_noun_target)

                        if 'MARGINAL' in criterion:
                            verb_loss += marginal_w * criterion['MARGINAL'](det_score, verb_target, 'verb')
                            noun_loss += marginal_w * criterion['MARGINAL'](det_score, noun_target, 'noun')

                        det_loss += v_w * verb_loss + n_w * noun_loss
                        losses_dict['det_cls'][phase] += det_loss.item() * batch_size
                        for i, fut_score in enumerate(fut_scores):
                            fut_score = fut_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            if i == 0:
                                fut_loss = f_w * criterion['MCE_EQL'](fut_score, fut_target)
                            else:
                                fut_loss += f_w * criterion['MCE_EQL'](fut_score, fut_target)
                        fut_verb_score, fut_noun_score = fut_verb_score.reshape(-1, fut_verb_score.shape[-1]), \
                            fut_noun_score.reshape(-1, fut_noun_score.shape[-1])
                        fut_verb_loss = criterion['MCE'](fut_verb_score, fut_verb_target)
                        fut_noun_loss = criterion['MCE'](fut_noun_score, fut_noun_target)

                        fut_loss += f_w * (fut_verb_loss + fut_noun_loss)
                        losses_dict['fut_cls'][phase] += fut_loss.item() * batch_size

                        det_loss += fut_loss

                    # Output log for current batch
                    pbar.set_postfix({
                        'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                        'det_loss': '{:.5f}'.format(det_loss.item()),
                    })

                    if training:
                        optimizer.zero_grad()
                        if det_loss.item() != 0:
                            det_loss.backward()
                            optimizer.step()
                            ema.update()
                            scheduler.step()
                    else:
                        global_start_idx = (batch_idx - 1) * batch_size

                        # Det scores (Action) - ONLY TAKE WORK MEMORY PART FOR DETECTION TARGETS
                        # Det scores (Action)
                        det_score_batch_np = det_scores[0][:, :cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES, :].softmax(
                            dim=-1).cpu().numpy()

                        current_work_steps = det_score_batch_np.shape[1]

                        # 获取 Target
                        # 先 reshape，再切片，最后 .cpu().numpy()
                        det_target_reshaped = det_target.reshape(batch_size, -1, cfg.DATA.NUM_CLASSES)
                        # 【关键】确保切片后立刻转 numpy
                        det_target_batch_np = det_target_reshaped[:, :current_work_steps, :].cpu().numpy()

                        # --- 2. 处理 Verb ---
                        verb_score_batch_np = verb_score[:, :cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES, :].softmax(dim=-1).cpu().numpy()
                        verb_target_reshaped = verb_target.reshape(batch_size, -1, verb_target.shape[-1])
                        verb_target_batch_np = verb_target_reshaped[:, :current_work_steps, :].cpu().numpy()

                        # --- 3. 处理 Noun ---
                        noun_score_batch_np = noun_score[:, :cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES, :].softmax(dim=-1).cpu().numpy()
                        noun_target_reshaped = noun_target.reshape(batch_size, -1, noun_target.shape[-1])
                        noun_target_batch_np = noun_target_reshaped[:, :current_work_steps, :].cpu().numpy()

                        for i in range(batch_size):
                            global_start_idx = (batch_idx - 1) * batch_size
                            global_idx = global_start_idx + i
                            if global_idx >= len(data_loaders[phase].dataset):  # Safety check
                                break
                            session_info = data_loaders[phase].dataset.inputs[global_idx]
                            session_name = session_info[
                                0]
                            pred_scores_by_session[session_name].append(det_score_batch_np[i])
                            gt_targets_by_session[session_name].append(det_target_batch_np[i])
                            pred_verb_scores_by_session[session_name].append(verb_score_batch_np[i])
                            gt_verb_targets_by_session[session_name].append(verb_target_batch_np[i])
                            pred_noun_scores_by_session[session_name].append(noun_score_batch_np[i])
                            gt_noun_targets_by_session[session_name].append(noun_target_batch_np[i])

        # --- END OF VALIDATION LOOP ---

        end = time.time()

        # Output log for current epoch
        log = []
        log.append('Epoch {:2}'.format(epoch))
        log.append('train det_loss: {:.5f}'.format(
            losses_dict['det_cls']['train'] / len(data_loaders['train'].dataset),
        ))

        if 'test' in cfg.SOLVER.PHASES and not training:  # Only evaluate if test phase exists and after the test loop
            # --- Concatenate predictions and targets for each session ---
            # The evaluation functions expect a dictionary where keys are session names
            # and values are numpy arrays of shape (num_frames_in_session, num_classes)
            final_pred_scores = {}
            final_gt_targets = {}
            final_pred_verb_scores = {}
            final_gt_verb_targets = {}
            final_pred_noun_scores = {}
            final_gt_noun_targets = {}

            # Iterate over all sessions collected
            for session_name in pred_scores_by_session:
                if pred_scores_by_session[session_name]:  # Check if list is not empty
                    # Concatenate list of (F, C) arrays along the frame dimension
                    final_pred_scores[session_name] = np.concatenate(pred_scores_by_session[session_name], axis=0)
                    final_gt_targets[session_name] = np.concatenate(gt_targets_by_session[session_name], axis=0)
                    final_pred_verb_scores[session_name] = np.concatenate(pred_verb_scores_by_session[session_name],
                                                                          axis=0)
                    final_gt_verb_targets[session_name] = np.concatenate(gt_verb_targets_by_session[session_name],
                                                                         axis=0)
                    final_pred_noun_scores[session_name] = np.concatenate(pred_noun_scores_by_session[session_name],
                                                                          axis=0)
                    final_gt_noun_targets[session_name] = np.concatenate(gt_noun_targets_by_session[session_name],
                                                                         axis=0)
                else:
                    # Handle empty sessions if any - unlikely but for safety
                    continue  # Skip empty sessions for now

            # --- Call the detailed evaluation function from new_metrics.py ---
            # Ensure the registry function name matches what's defined in new_metrics.py
            try:
                # This calls the function registered with @compute_result_new.register('EK100_final')
                eval_func = compute_result_new.get('EK100_final')
                if eval_func:
                    eval_func(cfg, final_pred_scores, final_gt_targets,
                              final_pred_verb_scores, final_gt_verb_targets,
                              final_pred_noun_scores, final_gt_noun_targets, logger)
                    import os
                    csv_filename = '{}_summary_final.csv'.format(cfg.DATA.DATA_NAME)
                    csv_absolute_path = os.path.abspath(csv_filename)
                    logger.info(f"Evaluation results saved to: {csv_absolute_path}")
                else:
                    logger.error("Evaluation function 'EK100_final' not found in registry.")
                    # Fallback to simple calculation if needed
                    # Calculate simple metrics using concatenated arrays if registry fails
                    all_gt = np.concatenate(list(final_gt_targets.values()), axis=0)
                    all_pred = np.concatenate(list(final_pred_scores.values()), axis=0)
                    action_labels = np.argmax(all_gt, axis=-1)
                    action_pred = all_pred.reshape(-1, 1, all_pred.shape[-1])
                    valid_index = list(np.where(action_labels != 0))[0]
                    if len(valid_index) > 0:
                        det_result = \
                        topk_recall_multiple_timesteps(action_pred[valid_index, ...], action_labels[valid_index], k=5)[
                            0]
                        log.append('test det_Recall: {:.5f}'.format(float(det_result) * 100))
                    else:
                        log.append('test det_Recall: N/A (no valid frames)')
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to simple calculation if needed
                all_gt = np.concatenate(list(final_gt_targets.values()), axis=0)
                all_pred = np.concatenate(list(final_pred_scores.values()), axis=0)
                action_labels = np.argmax(all_gt, axis=-1)
                action_pred = all_pred.reshape(-1, 1, all_pred.shape[-1])
                valid_index = list(np.where(action_labels != 0))[0]
                if len(valid_index) > 0:
                    det_result = \
                    topk_recall_multiple_timesteps(action_pred[valid_index, ...], action_labels[valid_index], k=5)[0]
                    log.append('test det_Recall (fallback): {:.5f}'.format(float(det_result) * 100))
                else:
                    log.append('test det_Recall (fallback): N/A (no valid frames)')

        log.append('running time: {:.2f} sec'.format(end - start))
        logger.info(' | '.join(log))

        # Save checkpoint for model and optimizer
        checkpointer.save(epoch, model, optimizer)
        if not training:
            ema.restore()
        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()
def main(cfg):
    #环境：seed，gpu or cpu，gpu_count
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='train')
    logger = setup_logger(cfg, phase='train')

    # Build data loaders
    # 实际执行流程：
    # 1. build_data_loader('train') 被调用
    # 2. build_dataset(cfg, 'train', '') 被调用
    # 3. DATA_LAYERS['THUMOS'] 返回 THUMOSDataset 类
    # 4. THUMOSDataset(cfg, 'train') 创建训练集实例
    # 5. DataLoader 包装训练集，设置 batch_size=32, shuffle=True 等
    data_loaders = {
        phase: build_data_loader(cfg, phase)
        for phase in cfg.SOLVER.PHASES
    }
    print(f'data_loaders:{data_loaders}')
    # Build model
    model = build_model(cfg, device)
    param_size = 0
    for param in model.parameters():
        # param.nelement(): 参数张量中的元素数量（浮点数个数）
        # 例如：形状为 [64, 128] 的权重矩阵有 64×128=8192 个元素

        # param.element_size(): 每个元素占用的字节数
        # 例如：float32 类型每个元素占 4 字节，float16 占 2 字节
        param_size += param.nelement() * param.element_size()
    logger.info('Number of parameters: {}. Model Size: {:.2f} MB'.format(sum(p.numel() for p in model.parameters()),
                                                                         param_size / 1024 ** 2))

    # Build criterion
    criterion = build_criterion(cfg, device)

    # Build optimizer
    optimizer = build_optimizer(cfg, model)

    # Build ema
    #平滑参数
    ema = build_ema(model, 0.999)

    # Load pretrained model and optimizer
    checkpointer.load(model, optimizer)

    # Build scheduler
    scheduler = build_scheduler(
        cfg, optimizer, len(data_loaders['train']))

    if cfg.DATA.DATA_NAME != 'EK100':
        do_perframe_det_train(cfg,
                              data_loaders,
                              model,
                              criterion,
                              optimizer,
                              scheduler,
                              ema,
                              device,
                              checkpointer,
                              logger)
    else:
        do_ek100_perframe_det_train(cfg,
                                    data_loaders,
                                    model,
                                    criterion,
                                    optimizer,
                                    scheduler,
                                    ema,
                                    device,
                                    checkpointer,
                                    logger)


######################################################

def do_perframe_det(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='StreamInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    # Collect detection scores and targets
    pred_scores = {}
    gt_targets = {}
    ant_scores = {}

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            target, _, _ = data[-4]
            score = model(*[x.to(device) for x in data[:-4]])
            score = score[0][-1].softmax(dim=-1).cpu().numpy()

            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        ant_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                    pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                if query_indices[0] in torch.arange(0, cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE):
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            full_indices = torch.cat((query_indices, torch.arange(
                                query_indices[-1] + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE,
                                query_indices[-1] + t_a + 1 + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE)), dim=0)
                            ant_scores[session][full_indices, :, t_a] = score[bs][:full_indices.shape[0]]

                    pred_scores[session][query_indices] = score[bs][:query_indices.shape[0]]
                    gt_targets[session][query_indices] = target[bs]
                else:
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            if query_indices[-1] + t_a + 1 < num_frames:
                                ant_scores[session][query_indices[-1] + t_a + 1, :, t_a] = score[bs][
                                    t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]

                    pred_scores[session][query_indices[-1]] = score[bs][-1 - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]
                    gt_targets[session][query_indices[-1]] = target[bs][-1]

        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
            maps_list = []
            for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                result = compute_result['perframe'](
                    cfg,
                    np.concatenate(list(gt_targets.values()), axis=0),
                    np.concatenate(list(ant_scores.values()), axis=0)[:, :, t_a],
                )
                logger.info('Action anticipation ({:.2f}s) perframe m{}: {:.5f}'.format(
                    (t_a + 1) / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE,
                    cfg.DATA.METRICS, result['mean_AP']
                ))
                maps_list.append(result['mean_AP'])
            logger.info('Action anticipation (mean) perframe m{}: {:.5f}'.format(
                cfg.DATA.METRICS, np.mean(maps_list)
            ))
        result = compute_result['perframe'](
            cfg,
            np.concatenate(list(gt_targets.values()), axis=0),
            np.concatenate(list(pred_scores.values()), axis=0),
        )
        logger.info('Action detection perframe m{}: {:.5f}'.format(
            cfg.DATA.METRICS, result['mean_AP']))

        # import os.path as osp
        # pkl.dump({'cfg': cfg,
        #           'scores': pred_scores,
        #           'labels': gt_targets,
        #           }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.detection.pkl', 'wb'))
def eval_perframe(cfg, ground_truth, prediction, class_names):
    from rekognition_online_action_detection.evaluation.metrics import perframe_average_precision
    ignore_index = cfg.DATA.IGNORE_INDEX
    metrics = cfg.DATA.METRICS
    postprocessing = default_pp(cfg.DATA.DATA_NAME)
    return perframe_average_precision(
        ground_truth=ground_truth,
        prediction=prediction,
        class_names=class_names,
        ignore_index=ignore_index,
        metrics=metrics,
        postprocessing=postprocessing,
    )
def compute_ek100_action_map_low_memory(
    cfg,
    gt_targets,
    pred_scores,
):
    """
    Compute EK100 action mAP class by class.

    This avoids concatenating the complete
    [num_frames, num_classes] prediction matrix.
    """
    sessions = list(pred_scores.keys())

    ignored_indices = {0}

    ignore_index = getattr(cfg.DATA, "IGNORE_INDEX", None)

    if isinstance(ignore_index, (list, tuple, set)):
        ignored_indices.update(
            int(index) for index in ignore_index
        )
    elif ignore_index is not None:
        ignored_indices.add(int(ignore_index))

    per_class_ap = []

    for class_index in range(cfg.DATA.NUM_CLASSES):
        if class_index in ignored_indices:
            continue

        gt_column = np.concatenate(
            [
                np.asarray(
                    gt_targets[session][:, class_index],
                    dtype=np.float32,
                )
                for session in sessions
            ],
            axis=0,
        )

        # Skip classes absent from the validation set.
        if not np.any(gt_column):
            del gt_column
            continue

        pred_column = np.concatenate(
            [
                np.asarray(
                    pred_scores[session][:, class_index],
                    dtype=np.float32,
                )
                for session in sessions
            ],
            axis=0,
        )

        class_ap = average_precision_score(
            gt_column,
            pred_column,
        )

        per_class_ap.append(class_ap)

        del gt_column
        del pred_column

    if len(per_class_ap) == 0:
        raise RuntimeError(
            "No valid EK100 action class was found "
            "during mAP evaluation."
        )

    return float(np.mean(per_class_ap))
def do_ek100_perframe_det(cfg, model, device, logger):
    # Setup model to test mode

    model.eval()
    cfg.MODEL.CRITERIONS = [['MCE', {}]]

    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='StreamInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    # Collect detection scores and targets
    pred_scores = {}
    gt_targets = {}
    gt_noun = {}
    gt_verb = {}
    pred_scores_verb = {}
    pred_scores_noun = {}
    anticipation_scores = {}
    anticipation_scores_verb = {}
    anticipation_scores_noun = {}
    cfg.DATA.NUM_VERBS = 126 if cfg.DATA.DATA_NAME == 'EK55' else 98
    cfg.DATA.NUM_NOUNS = 353 if cfg.DATA.DATA_NAME == 'EK55' else 301

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            if cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES > 0:
                total_target, _, = data[-4]
                target, noun_target, verb_target = total_target
                det_scores, _, noun_score, _, verb_score, _ = model(*[x.to(device) for x in data[:-4]])
                score = det_scores[-1].cpu().numpy()
                score_verb = verb_score.cpu().numpy()
                score_noun = noun_score.cpu().numpy()
            else:
                score, _, _ = model(*[x.to(device) for x in data[:-4]])
                target, noun_target, verb_target = data[-4]
                score = score.cpu().numpy()

            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        print(f'cfg.DATA.NUM_CLASSES:{cfg.DATA.NUM_CLASSES},num_frames:{num_frames},cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:{cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES}')
                        anticipation_scores[session] = np.zeros(
                            (num_frames, cfg.DATA.NUM_CLASSES, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                        if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                            anticipation_scores_verb[session] = np.zeros(
                                (num_frames, cfg.DATA.NUM_VERBS, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                            anticipation_scores_noun[session] = np.zeros(
                                (num_frames, cfg.DATA.NUM_NOUNS, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))

                    pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES),dtype=np.float32,
)
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        pred_scores_verb[session] = np.zeros((num_frames, cfg.DATA.NUM_VERBS))
                        pred_scores_noun[session] = np.zeros((num_frames, cfg.DATA.NUM_NOUNS))

                if session not in gt_targets:
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES),dtype=np.float32,
)
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        gt_noun[session] = np.zeros((num_frames, cfg.DATA.NUM_NOUNS))
                        gt_verb[session] = np.zeros((num_frames, cfg.DATA.NUM_VERBS))

                if query_indices[0] in torch.arange(0, cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE):
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            future_range = torch.arange(
                                query_indices[-1] + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE,
                                query_indices[-1] + t_a + 1 + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
                            )
                            full_indices = torch.cat((query_indices, future_range), dim=0)
                            valid_mask = full_indices < num_frames

                            if valid_mask.any():
                                valid_indices = full_indices[valid_mask]
                                num_valid = len(valid_indices)

                                # 取出对应的分数 (只取前 num_valid 个)
                                current_score = score[bs][:num_valid]

                                # 安全赋值
                                anticipation_scores[session][valid_indices, :, t_a] = current_score
                            if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                                anticipation_scores_noun[session][valid_indices, :, t_a] = score_noun[bs][:num_valid]
                                anticipation_scores_verb[session][valid_indices, :, t_a] = score_verb[bs][:num_valid]
                else:
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            if query_indices[-1] + t_a + 1 < num_frames:
                                anticipation_scores[session][query_indices[-1] + t_a + 1, :, t_a] = score[bs][
                                    t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]
                                if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                                    anticipation_scores_noun[session][query_indices[-1] + t_a + 1, :, t_a] = \
                                        score_noun[bs][t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]
                                    anticipation_scores_verb[session][query_indices[-1] + t_a + 1, :, t_a] = \
                                        score_verb[bs][t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]

                if query_indices[0] in torch.arange(0, cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE):
                    pred_scores[session][query_indices] = score[bs][:query_indices.shape[0]]
                    gt_targets[session][query_indices] = target[bs][:query_indices.shape[0]]
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        pred_scores_noun[session][query_indices] = score_noun[bs][:query_indices.shape[0]]
                        pred_scores_verb[session][query_indices] = score_verb[bs][:query_indices.shape[0]]
                        gt_noun[session][query_indices] = noun_target[bs][:query_indices.shape[0]]
                        gt_verb[session][query_indices] = verb_target[bs][:query_indices.shape[0]]
                else:
                    pred_scores[session][query_indices[-1]] = score[bs][-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES - 1]
                    gt_targets[session][query_indices[-1]] = target[bs][-1]
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        pred_scores_noun[session][query_indices[-1]] = score_noun[bs][
                            -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES - 1]
                        pred_scores_verb[session][query_indices[-1]] = score_verb[bs][
                            -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES - 1]
                        gt_noun[session][query_indices[-1]] = noun_target[bs][-1]
                        gt_verb[session][query_indices[-1]] = verb_target[bs][-1]

    # action_list = pd.read_csv('external/rulstm/RULSTM/data/ek100/actions.csv',
    #                            names=['id', 'verb', 'noun', 'action'], skipinitialspace=True, skiprows=1)
    # action_names, verb_names, noun_names = {'Background': 0}, {'Background': 0}, {'Background': 0}
    # predictions_verb, predictions_noun = [], []
    # discarded_labels, discarded_ids = [], []
    # for acts in action_list.iterrows():
    #     act = acts[1]
    #     action_names[act['action']] = act['id'] + 1
    #     tmp = act['action'].split(' ')
    #     verb, noun = tmp[0], ' '.join(tmp[1:])
    #     verb_names[verb] = act['verb'] + 1
    #     noun_names[noun] = act['noun'] + 1

    map_result_act = eval_perframe(cfg, np.concatenate(list(gt_targets.values()), axis=0),
                                   np.concatenate(list(pred_scores.values()), axis=0), cfg.DATA.CLASS_NAMES)['mean_AP']

    map_result_noun = eval_perframe(cfg, np.concatenate(list(gt_noun.values()), axis=0),
                                    np.concatenate(list(pred_scores_noun.values()), axis=0),
                                    [str(i) for i in range(301)])['mean_AP']

    map_result_verb = eval_perframe(cfg, np.concatenate(list(gt_verb.values()), axis=0),
                                    np.concatenate(list(pred_scores_verb.values()), axis=0),
                                    [str(i) for i in range(98)])['mean_AP']

    oad_score, oad_noun_score, oad_verb_score = np.concatenate(list(pred_scores.values()), axis=0), \
        np.concatenate(list(pred_scores_noun.values()), axis=0), \
        np.concatenate(list(pred_scores_verb.values()), axis=0)
    oad_target, oad_noun_target, oad_verb_target = np.concatenate(list(gt_targets.values()), axis=0), \
        np.concatenate(list(gt_noun.values()), axis=0), \
        np.concatenate(list(gt_verb.values()), axis=0)

    action_labels, noun_labels, verb_labels = np.argmax(oad_target, axis=-1), \
        np.argmax(oad_noun_target, axis=-1), np.argmax(oad_verb_target, axis=-1)
    action_pred, oad_noun_pred, oad_verb_pred = oad_score.reshape(-1, 1, oad_score.shape[-1]), \
        oad_noun_score.reshape(-1, 1, oad_noun_score.shape[-1]), \
        oad_verb_score.reshape(-1, 1, oad_verb_score.shape[-1])
    valid_index = list(np.where(action_labels != 0))[0]
    overall_act_res = topk_recall_multiple_timesteps(action_pred[valid_index, ...], action_labels[valid_index], k=5)[0]
    overall_noun_res = topk_recall_multiple_timesteps(oad_noun_pred[valid_index, ...], noun_labels[valid_index], k=5)[0]
    overall_verb_res = topk_recall_multiple_timesteps(oad_verb_pred[valid_index, ...], verb_labels[valid_index], k=5)[0]

    logger.info(
        f"Online Detection Results: Overall Action: {float(overall_act_res) * 100:.6f}, Verb: {float(overall_verb_res) * 100:.6f}, Noun: {float(overall_noun_res) * 100:.6f}")

    ## for aticipation
    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
        path_to_data = 'external/rulstm/RULSTM/data/ek100/'
        segment_list = pd.read_csv(osp.join(path_to_data, 'validation.csv'),
                                   names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                   skipinitialspace=True)
        predictions, labels, ids = [], [], []
        predictions_verb, predictions_noun = [], []
        discarded_labels, discarded_ids = [], []
        for segment in segment_list.iterrows():
            start_step = int(np.floor(segment[1]['start_f'] / 30 * cfg.DATA.FPS))
            vid = segment[1]['video'].strip()
            preds = (anticipation_scores[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
            preds_v = (anticipation_scores_verb[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
            preds_n = (anticipation_scores_noun[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
            if start_step >= cfg.MODEL.LSTR.ANTICIPATION_LENGTH:
                predictions.append(preds)
                if preds_v is not None:
                    predictions_verb.append(preds_v)
                if preds_n is not None:
                    predictions_noun.append(preds_n)
                labels.append(segment[1][['verb', 'noun', 'action']].values.astype(int))
                ids.append(segment[1]['id'])
            else:
                discarded_labels.append(segment[1][['verb', 'noun', 'action']].values.astype(int))
                discarded_ids.append(segment[1]['id'])

        actions = pd.read_csv(osp.join(path_to_data, 'actions.csv'), index_col='id')

        action_scores = np.concatenate(predictions)
        if len(predictions_verb) > 0:
            verb_scores = np.concatenate(predictions_verb)
        else:
            verb_scores = None
        if len(predictions_noun) > 0:
            noun_scores = np.concatenate(predictions_noun)
        else:
            noun_scores = None
        labels = np.array(labels)
        ids = np.array(ids)

        noun_scores = None
        verb_scores = None

        vi = get_marginal_indexes(actions, 'verb')
        ni = get_marginal_indexes(actions, 'noun')

        action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))
        if verb_scores is None:
            verb_scores = marginalize(action_probs, vi).reshape(action_scores.shape[0],
                                                                action_scores.shape[1],
                                                                -1)
        else:
            verb_scores = softmax(verb_scores.reshape(-1, verb_scores.shape[-1]))
            verb_scores = verb_scores.reshape(action_scores.shape[0],
                                              action_scores.shape[1],
                                              -1)
        if noun_scores is None:
            noun_scores = marginalize(action_probs, ni).reshape(action_scores.shape[0],
                                                                action_scores.shape[1],
                                                                -1)
        else:
            noun_scores = softmax(noun_scores.reshape(-1, noun_scores.shape[-1]))
            noun_scores = noun_scores.reshape(action_scores.shape[0],
                                              action_scores.shape[1],
                                              -1)
        dlab = np.array(discarded_labels)
        dislab = np.array(discarded_ids)
        ids = np.concatenate([ids, dislab])
        num_disc = len(dlab)
        labels = np.concatenate([labels, dlab])
        verb_scores = np.concatenate((verb_scores, np.zeros((num_disc, *verb_scores.shape[1:]))))
        noun_scores = np.concatenate((noun_scores, np.zeros((num_disc, *noun_scores.shape[1:]))))
        action_scores = np.concatenate((action_scores, np.zeros((num_disc, *action_scores.shape[1:]))))

        verb_labels, noun_labels, action_labels = labels[:, 0], labels[:, 1], labels[:, 2]

        overall_verb_recalls = topk_recall_multiple_timesteps(verb_scores, verb_labels, k=5)
        overall_noun_recalls = topk_recall_multiple_timesteps(noun_scores, noun_labels, k=5)
        overall_action_recalls = topk_recall_multiple_timesteps(action_scores, action_labels, k=5)

        unseen = pd.read_csv(osp.join(path_to_data, 'validation_unseen_participants_ids.csv'), names=['id']).squeeze(
            "columns")
        tail_verbs = pd.read_csv(osp.join(path_to_data, 'validation_tail_verbs_ids.csv'), names=['id']).squeeze(
            "columns")
        tail_nouns = pd.read_csv(osp.join(path_to_data, 'validation_tail_nouns_ids.csv'), names=['id']).squeeze(
            "columns")
        tail_actions = pd.read_csv(osp.join(path_to_data, 'validation_tail_actions_ids.csv'), names=['id']).squeeze(
            "columns")

        unseen_bool_idx = pd.Series(ids).isin(unseen).values
        tail_verbs_bool_idx = pd.Series(ids).isin(tail_verbs).values
        tail_nouns_bool_idx = pd.Series(ids).isin(tail_nouns).values
        tail_actions_bool_idx = pd.Series(ids).isin(tail_actions).values

        tail_verb_recalls = topk_recall_multiple_timesteps(
            verb_scores[tail_verbs_bool_idx], verb_labels[tail_verbs_bool_idx], k=5)
        tail_noun_recalls = topk_recall_multiple_timesteps(
            noun_scores[tail_nouns_bool_idx], noun_labels[tail_nouns_bool_idx], k=5)
        tail_action_recalls = topk_recall_multiple_timesteps(
            action_scores[tail_actions_bool_idx], action_labels[tail_actions_bool_idx], k=5)

        unseen_verb_recalls = topk_recall_multiple_timesteps(
            verb_scores[unseen_bool_idx], verb_labels[unseen_bool_idx], k=5)
        unseen_noun_recalls = topk_recall_multiple_timesteps(
            noun_scores[unseen_bool_idx], noun_labels[unseen_bool_idx], k=5)
        unseen_action_recalls = topk_recall_multiple_timesteps(
            action_scores[unseen_bool_idx], action_labels[unseen_bool_idx], k=5)

        all_accuracies = np.concatenate(
            [overall_verb_recalls, overall_noun_recalls, overall_action_recalls,
             unseen_verb_recalls, unseen_noun_recalls, unseen_action_recalls,
             tail_verb_recalls, tail_noun_recalls, tail_action_recalls]
        )  # 9 x 8

        indices = [
            ('Overall Mean Top-5 Recall', 'Verb'), ('Overall Mean Top-5 Recall', 'Noun'),
            ('Overall Mean Top-5 Recall', 'Action'),
            ('Unseen Mean Top-5 Recall', 'Verb'), ('Unseen Mean Top-5 Recall', 'Noun'),
            ('Unseen Mean Top-5 Recall', 'Action'),
            ('Tail Mean Top-5 Recall', 'Verb'), ('Tail Mean Top-5 Recall', 'Noun'),
            ('Tail Mean Top-5 Recall', 'Action'),
        ]

        cc = np.linspace(1 / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_LENGTH,
                         1 / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE,
                         cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES, dtype=str)
        scores = pd.DataFrame(all_accuracies * 100, columns=cc,
                              index=pd.MultiIndex.from_tuples(indices))
        logger.info(scores)

        result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                      'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                      'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                      'mAP_verb': float(map_result_verb) * 100,
                      'map_noun': float(map_result_noun) * 100,
                      'mAP_act': float(map_result_act) * 100,
                      'top5_Rec_verb': float(overall_verb_res) * 100,
                      'top5_Rec_noun': float(overall_noun_res) * 100,
                      'top5_Rec_action': float(overall_act_res) * 100,
                      '||Anticp|| ': ' || ',
                      'ant_t5Rec_verb': overall_verb_recalls * 100,
                      'ant_t5Rec_noun': overall_noun_recalls * 100,
                      'ant_t5Rec_action': overall_action_recalls * 100,
                      'ant_t5Rec_unseen_v': unseen_verb_recalls * 100,
                      'ant_t5Rec_unseen_n': unseen_noun_recalls * 100,
                      'ant_t5Rec_unseen_a': unseen_action_recalls * 100,
                      'ant_t5Rec_tail_v': tail_verb_recalls * 100,
                      'ant_t5Rec_tail_n': tail_noun_recalls * 100,
                      'ant_t5Rec_tail_a': tail_action_recalls * 100,
                      }
    else:
        result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                      'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                      'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                      'mAP_verb': float(map_result_verb) * 100,
                      'map_noun': float(map_result_noun) * 100,
                      'mAP_act': float(map_result_act) * 100,
                      'top5_Rec_verb': float(overall_verb_res) * 100,
                      'top5_Rec_noun': float(overall_noun_res) * 100,
                      'top5_Rec_action': float(overall_act_res) * 100,
                      '||Anticp|| ': ' || ',
                      'ant_t5Rec_verb': '-',
                      'ant_t5Rec_noun': '-',
                      'ant_t5Rec_action': '-',
                      'ant_t5Rec_unseen_v': '-',
                      'ant_t5Rec_unseen_n': '-',
                      'ant_t5Rec_unseen_a': '-',
                      'ant_t5Rec_tail_v': '-',
                      'ant_t5Rec_tail_n': '-',
                      'ant_t5Rec_tail_a': '-',
                      }

    df = pd.DataFrame([result_all])
    import os
    head = True
    if os.path.exists('{}_summary.csv'.format(cfg.DATA.DATA_NAME)):
        head = False
    df.to_csv('{}_summary.csv'.format(cfg.DATA.DATA_NAME), mode='a', index=False, header=head, float_format='%.1f')


def infer(cfg):
    # Setup configurations

    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    logger = setup_logger(cfg, phase='test')

    # Build model
    model = build_model(cfg, device)

    # Load pretrained model
    checkpointer.load(model)
    if cfg.DATA.DATA_NAME != 'EK100':
        do_perframe_det(
            cfg,
            model,
            device,
            logger,
        )
    else:

        print(f"gpu:{torch.cuda.is_available()}")
        # 将模型移动到 GPU
        model = model.to(device)
        do_ek100_perframe_det(
            cfg,
            model,
            device,
            logger,
        )


if __name__ == '__main__':
    cfg = load_cfg()
    if not cfg.TEST:
        main(cfg)
    else:
        infer(cfg)

