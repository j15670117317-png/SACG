# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from bisect import bisect_right

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import copy

from .datasets import DATA_LAYERS as registry
#from rekognition_online_action_detection.utils.ek_utils import (action_to_noun_map, action_to_verb_map)


@registry.register('THUMOS')
@registry.register('CrossTask')
@registry.register('TVSeries')
class CMeRTBatchDataLayer(data.Dataset):
    # work feature, (det,ant_target, fut_target) where fut_target is from work_start
    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.data_name = cfg.DATA.DATA_NAME
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_num_samples = self.anticipation_length // cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        # Future choice
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
        # context length
        self.context_length = int(cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS)
        self.context_sample_rate = cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE
        self.context_num_samples = int(cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS) // cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE

        self.training = phase == 'train'

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        self.inputs = []
        if self.data_name.startswith('THUMOS') or self.data_name.startswith('CrossTask') or self.data_name.startswith('TVSeries'):
            for session in self.sessions:
                target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
                seed = np.random.randint(self.work_memory_length) if self.training else 0
                for work_start, ant_end in zip(range(seed, target.shape[0], self.work_memory_length),
                                               range(seed + self.work_memory_length + self.anticipation_length,
                                                     target.shape[0] - self.anticipation_length, self.work_memory_length)):
                    self.inputs.append([session, work_start, ant_end, target, None, ])

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        if start < 0:
            start = (end + 1) % sample_rate
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        (session, work_start, ant_end, target, segments_before) = self.inputs[index]
        visual_inputs = np.load(osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get work memory
        work_end = ant_end - self.anticipation_length
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]

        #context_indices = np.arange(work_start-self.context_num_samples,  work_start).clip(0)
        context_indices = np.arange(work_start - self.context_length, work_start).clip(0)
        context_indices = context_indices[::self.context_sample_rate]
        work_indices = np.concatenate((context_indices, work_indices))

        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        # Get target
        total_target = copy.deepcopy(target)
        target = target[work_start: ant_end][::self.work_memory_sample_rate]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = work_start - self.long_memory_length, work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get future
        if self.future_num_samples > 0:
            future_start, future_end = min(work_start, total_target.shape[0] - 1), min(work_start + self.future_num_samples - 1, total_target.shape[0] - 1)
            future_indices = self.future_sampler(future_start, future_end, self.future_num_samples, self.future_sample_rate)
            future_target = total_target[future_indices]
        else:
            future_target = None

        # Get all memory
        if (long_visual_inputs is not None and long_motion_inputs is not None):
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))
        if future_target is not None:
            future_target = torch.as_tensor(future_target.astype(np.float32))
        else:
            future_target = target
        if self.anticipation_num_samples > 0:
            target = (target[:self.work_memory_num_samples], target[self.work_memory_num_samples:], future_target)
        else:
            target = (target[:self.work_memory_num_samples], target, future_target)

        assert fusion_visual_inputs.ndim == fusion_motion_inputs.ndim

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, fusion_motion_inputs, target)

    def __len__(self):
        return len(self.inputs)

@registry.register('EK100')
class LSTRDataLayer_(data.Dataset):

    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        # Anticipation choice
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        # Future choice
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES

        self.training = phase == 'train'
        self.clip_mixup_rate = 0.5
        self.clip_mixup_sample = 'uniform'
        self.plus_ratio = 0.25
        self.rgb = dict()
        self.flow = dict()
        self.noun = dict()
        self.target = dict()
        self.verb = dict()
        self.cfg = cfg
        self.data_name = cfg.DATA.DATA_NAME
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_num_samples = self.anticipation_length // cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        # Future choice
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
        # context length
        self.context_length = int(cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS)
        self.context_sample_rate = cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE
        self.context_num_samples = int(
            cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS) // cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        path_to_data = 'external/rulstm/RULSTM/data/ek100/'
        segment_list = pd.read_csv(osp.join(path_to_data,
                                            'training.csv' if self.training else 'validation.csv'),
                                   names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                   skipinitialspace=True)
        segment_list['len_f'] = segment_list['end_f'] - segment_list['start_f']

        self.segment_by_cls = {cls: segment_list[segment_list['action'] == cls]
                               for cls in range(0, self.cfg.DATA.NUM_CLASSES - 1)}
        self.inputs = []
        for session in self.sessions:
            # target_path = osp.join(self.data_root, self.target_perframe, session + '.npy')
            # target = np.load(target_path, mmap_mode='r')
            # verb_target = np.load(osp.join(self.data_root, self.target_perframe.replace('target', 'verb'), session + '.npy'))
            # noun_target = np.load(osp.join(self.data_root, self.target_perframe.replace('target', 'noun'), session + '.npy'))
            paths = {
                'target': osp.join(self.data_root, self.target_perframe, session + '.npy'),
                'verb': osp.join(self.data_root, self.target_perframe.replace('target', 'verb'), session + '.npy'),
                'noun': osp.join(self.data_root, self.target_perframe.replace('target', 'noun'), session + '.npy'),
                'rgb': osp.join(self.data_root, self.visual_feature, session + '.npy'),
                'flow': osp.join(self.data_root, self.motion_feature, session + '.npy')
            }
            self.rgb[session] =  paths['rgb']
            self.flow[session] = paths['flow']
            self.target[session] = paths['target']
            self.noun[session] = paths['noun']
            self.verb[session] = paths['verb']

            segments_per_session = segment_list[segment_list['video'] == session]
            for segment in segments_per_session.iterrows():
                start_tick = int(segment[1]['start_f'] / 30 * self.cfg.DATA.FPS)
                end_tick = int(segment[1]['end_f'] / 30 * self.cfg.DATA.FPS)
                start_tick += np.random.randint(self.anticipation_length) if self.training else 0
                start_tick = min(start_tick, end_tick)
                work_end = start_tick - self.anticipation_length
                work_start = work_end - self.work_memory_length
                segments_before_current = segments_per_session[segments_per_session['end_f'] < segment[1]['start_f']]
                segments_before_end = segments_per_session[segments_per_session['end_f'] < segment[1]['end_f']]

                if work_start < 0:
                    continue
                self.inputs.append([
                    session, work_start, work_end,
                    paths['target'], paths['verb'], paths['noun'],  # 存路径
                    segments_before_end,
                    segments_before_current
                ])

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        # session, work_start, work_end, target = self.inputs[index]
        session, work_start, work_end, target, verb_target, noun_target, segments_work, segments_before = self.inputs[index]

        visual_inputs = np.load(self.rgb[session], mmap_mode='r')
        motion_inputs = np.load(self.flow[session], mmap_mode='r')

        target = np.load(target, mmap_mode='r')
        verb_target = np.load(verb_target, mmap_mode='r')
        noun_target = np.load(noun_target, mmap_mode='r')


        # Get target
        total_target = target
        total_noun_target = noun_target
        total_verb_target = verb_target
        # target = target[work_start: work_end][::self.work_memory_sample_rate]

        # Get work memory
        if self.future_num_samples > 0:
            target = target[work_start: work_end + self.anticipation_length]
            noun_target = noun_target[work_start: work_end + self.anticipation_length]
            verb_target = verb_target[work_start: work_end + self.anticipation_length]
            target = np.concatenate((target[:self.work_memory_length:self.work_memory_sample_rate],
                                     target[self.work_memory_length::self.anticipation_sample_rate]),
                                    axis=0)
            noun_target = np.concatenate((noun_target[:self.work_memory_length:self.work_memory_sample_rate],
                                          noun_target[self.work_memory_length::self.anticipation_sample_rate]),
                                         axis=0)
            verb_target = np.concatenate((verb_target[:self.work_memory_length:self.work_memory_sample_rate],
                                          verb_target[self.work_memory_length::self.anticipation_sample_rate]),
                                         axis=0)
            # Get work memory
            work_indices = np.arange(work_start, work_end).clip(0)
            work_indices = work_indices[::self.work_memory_sample_rate]

            # context_indices = np.arange(work_start-self.context_num_samples,  work_start).clip(0)
            context_indices = np.arange(work_start - self.context_length, work_start).clip(0)
            context_indices = context_indices[::self.context_sample_rate]
            work_indices = np.concatenate((context_indices, work_indices))

            work_visual_inputs = visual_inputs[work_indices]
            work_motion_inputs = motion_inputs[work_indices]
        else:
            target = target[work_start: work_end][::self.work_memory_sample_rate]
            noun_target = noun_target[work_start: work_end][::self.work_memory_sample_rate]
            verb_target = verb_target[work_start: work_end][::self.work_memory_sample_rate]
            work_indices = np.arange(work_start, work_end).clip(0)
            work_indices = work_indices[::self.work_memory_sample_rate]
            work_visual_inputs = visual_inputs[work_indices]
            work_motion_inputs = motion_inputs[work_indices]

        # Mixclip++
        if self.training and self.clip_mixup_rate > 0:
            assert segments_work is not None
            segments_work = segments_work[segments_work['end_f'] / 30 * self.cfg.DATA.FPS > work_start]
            prob = None
            num_clip = int(len(segments_work) * 0.5)
            segments_to_mix = segments_work.sample(num_clip, replace=False,
                                                   weights=prob)
            for segment in segments_to_mix.iterrows():
                start, end = int(segment[1]['start_f'] / 30 * self.cfg.DATA.FPS), int(np.ceil(segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                random_cls = np.random.randint(0, self.cfg.DATA.NUM_CLASSES - 1)
                vid = segment[1]['video']
                # random_cls = segment[1]['action']
                assert vid == session
                segment_random_cls = self.segment_by_cls[random_cls]
                segment_random_cls = segment_random_cls[segment_random_cls['video'] != vid]
                valid_segments = segment_random_cls[segment_random_cls['len_f'] > segment[1]['len_f']]
                if len(valid_segments) == 0:
                    continue
                else:
                    sample_segment = valid_segments.sample(1)
                    new_vid = sample_segment['video'].values[0]
                    new_start_tick = int((sample_segment['start_f'].iloc[0] / 30) * self.cfg.DATA.FPS)
                    new_end_tick = int((sample_segment['end_f'].iloc[0] / 30) * self.cfg.DATA.FPS)

                    new_visual_inputs = np.load(self.rgb[new_vid], mmap_mode='r')
                    new_motion_inputs = np.load(self.flow[new_vid], mmap_mode='r')
                    # 如果需要 target
                    new_target = np.load(self.target[new_vid], mmap_mode='r')
                    new_noun = np.load(self.noun[new_vid], mmap_mode='r')
                    new_verb = np.load(self.verb[new_vid], mmap_mode='r')

                    sel_indices = np.where((work_indices >= start) & (work_indices <= end))
                    shift = np.random.randint(new_end_tick - new_start_tick - len(sel_indices) + 1)

                    work_visual_inputs[sel_indices] = self.plus_ratio * new_visual_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)] + (1 - self.plus_ratio) * \
                                                      work_visual_inputs[sel_indices]
                    work_motion_inputs[sel_indices] = self.plus_ratio * new_motion_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)] + (1 - self.plus_ratio) * \
                                                      work_motion_inputs[sel_indices]
                    target[sel_indices] = self.plus_ratio * new_target[new_start_tick + shift:new_start_tick + shift + len(sel_indices)] + (1 - self.plus_ratio) * target[sel_indices]
                    verb_target[sel_indices] = self.plus_ratio * new_verb[new_start_tick + shift:new_start_tick + shift + len(sel_indices)] + (1 - self.plus_ratio) * verb_target[sel_indices]
                    noun_target[sel_indices] = self.plus_ratio * new_noun[new_start_tick + shift:new_start_tick + shift + len(sel_indices)] + (1 - self.plus_ratio) * noun_target[sel_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
                # memory_key_padding_mask[:last_zero] = -1e5

            # Mixclip++
            if self.training and self.clip_mixup_rate > 0:
                assert segments_before is not None
                segments_before = segments_before[segments_before['end_f'] / 30 * self.cfg.DATA.FPS > long_start]
                if self.clip_mixup_sample == 'uniform':
                    prob = None
                elif self.clip_mixup_sample == 'by_length':
                    prob = (segments_before['len_f']).tolist()
                    prob = prob / np.sum(prob)
                else:
                    raise ValueError
                num_clip = int(len(segments_before) * self.clip_mixup_rate)
                segments_to_mixup = segments_before.sample(num_clip, replace=False,
                                                           weights=prob)
                for old_segment in segments_to_mixup.iterrows():
                    old_start_tick = int(old_segment[1]['start_f'] / 30 * self.cfg.DATA.FPS)
                    old_end_tick = int(np.ceil(old_segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                    old_action = old_segment[1]['action']
                    # old_action = np.random.randint(0, self.cfg.DATA.NUM_CLASSES - 1)
                    old_vid = old_segment[1]['video']
                    assert old_vid == session
                    segment_same_class = self.segment_by_cls[old_action]
                    segment_same_class = segment_same_class[segment_same_class['video'] != old_vid]
                    valid_segments = segment_same_class[segment_same_class['len_f'] > old_segment[1]['len_f']]
                    if len(valid_segments) == 0:
                        continue
                    else:
                        sample_segment = valid_segments.sample(1)
                        new_vid = sample_segment['video'].values[0]
                        new_start_tick = int((sample_segment['start_f'].iloc[0] / 30) * self.cfg.DATA.FPS)
                        new_end_tick = int((sample_segment['end_f'].iloc[0] / 30) * self.cfg.DATA.FPS)

                        new_visual_inputs = np.load(self.rgb[new_vid], mmap_mode='r')
                        new_motion_inputs = np.load(self.flow[new_vid], mmap_mode='r')


                        sel_indices = np.where((long_indices >= old_start_tick) & (long_indices <= old_end_tick))

                        shift = np.random.randint(max(new_end_tick - new_start_tick - len(sel_indices) + 1, 1))
                        long_visual_inputs[sel_indices] = new_visual_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)]
                        long_motion_inputs[sel_indices] = new_motion_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)]

        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None


        # Get future
        if self.future_num_samples > 0:
            future_start, future_end = min(work_start + 1, total_target.shape[0] - 1), min(work_start + self.future_num_samples - 1, total_target.shape[0] - 1)
            future_indices = self.future_sampler(future_start, future_end, self.future_num_samples, self.future_sample_rate)
            future_target = total_target[future_indices]
            future_noun_target = total_noun_target[future_indices]
            future_verb_target = total_verb_target[future_indices]
        else:
            future_target = None
            future_noun_target = None
            future_verb_target = None


        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        n_det = self.work_memory_num_samples
        # 如果开启了 anticipation，target 的后半部分是 ant，否则 ant 为空
        if self.anticipation_num_samples > 0:
            det_target_action = target[:n_det]
            ant_target_action = target[n_det:]

            det_target_verb = verb_target[:n_det]
            ant_target_verb = verb_target[n_det:]

            det_target_noun = noun_target[:n_det]
            ant_target_noun = noun_target[n_det:]
        else:
            det_target_action = target
            ant_target_action = torch.zeros(0, target.shape[1])  # 空 Tensor

            det_target_verb = verb_target
            ant_target_verb = torch.zeros(0, verb_target.shape[1])

            det_target_noun = noun_target
            ant_target_noun = torch.zeros(0, noun_target.shape[1])

        # 2. 处理 Future (如果有)
        if future_target is not None:
            fut_target_action = future_target
            fut_target_verb = future_verb_target
            fut_target_noun = future_noun_target
        else:
            # 如果没有 future，创建空 Tensor 占位，保持结构一致
            c_act = target.shape[1]
            c_verb = verb_target.shape[1]
            c_noun = noun_target.shape[1]
            fut_target_action = torch.zeros(0, c_act)
            fut_target_verb = torch.zeros(0, c_verb)
            fut_target_noun = torch.zeros(0, c_noun)

        # 3. 转换为 Tensor (如果还没转)
        # 前面的切片操作可能返回 numpy 或 tensor，统一转为 tensor
        det_target_action = torch.as_tensor(
            det_target_action.astype(np.float32) if isinstance(det_target_action, np.ndarray) else det_target_action)
        ant_target_action = torch.as_tensor(
            ant_target_action.astype(np.float32) if isinstance(ant_target_action, np.ndarray) else ant_target_action)
        fut_target_action = torch.as_tensor(
            fut_target_action.astype(np.float32) if isinstance(fut_target_action, np.ndarray) else fut_target_action)

        det_target_verb = torch.as_tensor(
            det_target_verb.astype(np.float32) if isinstance(det_target_verb, np.ndarray) else det_target_verb)
        ant_target_verb = torch.as_tensor(
            ant_target_verb.astype(np.float32) if isinstance(ant_target_verb, np.ndarray) else ant_target_verb)
        fut_target_verb = torch.as_tensor(
            fut_target_verb.astype(np.float32) if isinstance(fut_target_verb, np.ndarray) else fut_target_verb)

        det_target_noun = torch.as_tensor(
            det_target_noun.astype(np.float32) if isinstance(det_target_noun, np.ndarray) else det_target_noun)
        ant_target_noun = torch.as_tensor(
            ant_target_noun.astype(np.float32) if isinstance(ant_target_noun, np.ndarray) else ant_target_noun)
        fut_target_noun = torch.as_tensor(
            fut_target_noun.astype(np.float32) if isinstance(fut_target_noun, np.ndarray) else fut_target_noun)

        # 4. 构造最终的 Targets 元组 (9 个元素)
        # 顺序: det_act, ant_act, fut_act, det_verb, ant_verb, fut_verb, det_noun, ant_noun, fut_noun
        targets_tuple = (
            det_target_action, ant_target_action, fut_target_action,
            det_target_verb, ant_target_verb, fut_target_verb,
            det_target_noun, ant_target_noun, fut_target_noun
        )

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, targets_tuple
        else:
            return fusion_visual_inputs, fusion_motion_inputs, targets_tuple
    def __len__(self):
        return len(self.inputs)



@registry.register('StreamInferenceTHUMOS')
@registry.register('StreamInferenceCrossTask')
@registry.register('StreamInferenceTVSeries')
class CMeRTStreamInferenceDataLayer(data.Dataset):
    # work feature, det_target, work_index

    def __init__(self, cfg, phase='test'):
        self.cfg = cfg
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.object_feature = cfg.INPUT.OBJECT_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.anticipation_num_samples = self.anticipation_length // self.anticipation_sample_rate
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
        self.context_length = int(cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS)
        self.context_sample_rate = cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE
        self.context_num_samples = int(
            cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS) // cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            for work_start, work_end in zip(
                    range(0, target.shape[0] + 1),
                    range(self.work_memory_length, target.shape[0] + 1)):
                self.inputs.append([
                    session, work_start, work_end,
                    target,
                    target.shape[0]
                ])

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        if start < 0:
            start = (end + 1) % sample_rate
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target, num_frames = self.inputs[index]
        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')


        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        # Get target
        total_target = copy.deepcopy(target)
        target = target[work_indices]

        context_indices = np.arange(work_start - self.context_length, work_start).clip(0)
        context_indices = context_indices[::self.context_sample_rate]
        all_indices = np.concatenate((context_indices, work_indices))
        work_visual_inputs = visual_inputs[all_indices]
        work_motion_inputs = motion_inputs[all_indices]



        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = work_start - self.long_memory_length, work_start - 1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        future_target = None

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        if future_target is not None:
            future_target = torch.as_tensor(future_target.astype(np.float32))
        else:
            future_target = target
        target = torch.as_tensor(target.astype(np.float32))
        target = (target, target, future_target)

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target,
                    session, work_indices, num_frames)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, fusion_motion_inputs, target,
                    session, work_indices, num_frames)

    def __len__(self):
        return len(self.inputs)

@registry.register('StreamInferenceEK100')
class LSTRBatchInferenceDataLayer(data.Dataset):

    def __init__(self, cfg, phase='test'):
        self.cfg = cfg
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME

        # 保存 target 相关的路径模板，而不是加载数据
        self.target_path_template = osp.join(self.data_root, self.target_perframe, '{}.npy')

        if self.cfg.MODEL.LSTR.V_N_CLASSIFIER:
            self.verb_path_template = osp.join(self.data_root, self.target_perframe.replace('target', 'verb'), '{}.npy')
            self.noun_path_template = osp.join(self.data_root, self.target_perframe.replace('target', 'noun'), '{}.npy')
        else:
            self.verb_path_template = None
            self.noun_path_template = None

        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES

        assert phase == 'test', 'phase must be `test` for batch inference, got {}'

        self.inputs = []
        # 【修改点 1】：初始化阶段只计算索引范围，不加载任何 npy 文件
        for session in self.sessions:
            # 只需要知道总帧数即可，这里为了获取 num_frames 暂时需要加载一次 shape
            # 如果连 shape 都不想加载，可以预先存一个 json 记录每个 session 的帧数
            # 但通常加载一个小的 label 文件的 shape 不会爆内存，爆内存是因为加载了所有数据内容
            # 为了极致优化，也可以把 num_frames 也做成懒加载，或者预先存在列表里
            temp_target_path = self.target_path_template.format(session)
            # 使用 mmap 快速读取 shape，不加载具体内容到内存
            temp_target = np.load(temp_target_path, mmap_mode='r')
            num_frames = temp_target.shape[0]
            del temp_target  # 立即释放

            for work_start, work_end in zip(
                    range(0, num_frames + self.work_memory_sample_rate),
                    range(self.work_memory_length, num_frames + self.work_memory_sample_rate)):
                # 【修改点 2】：self.inputs 中只存 session 名称和起止索引，不存数组对象
                self.inputs.append([
                    session,
                    work_start,
                    work_end,
                    num_frames
                ])

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        if start < 0:
            start = (end + 1) % sample_rate
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        # 【修改点 3】：从 self.inputs 取出基本信息
        session, work_start, work_end, num_frames = self.inputs[index]

        # 【修改点 4】：在 __getitem__ 内部按需加载所有 .npy 文件 (Lazy Load)
        # 这样每个进程每次只加载当前需要的几个文件，用完即释
        target = np.load(self.target_path_template.format(session), mmap_mode='r')

        if self.cfg.MODEL.LSTR.V_N_CLASSIFIER:
            verb_target = np.load(self.verb_path_template.format(session), mmap_mode='r')
            noun_target = np.load(self.noun_path_template.format(session), mmap_mode='r')
        else:
            verb_target = target
            noun_target = target

        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get target
        # 注意：切片操作会创建新数组，但原数组 target 在函数结束后会被回收
        total_target = target  # 不需要 deepcopy，mmap 对象本身不可写或我们只读
        target_slice = target[work_start: work_end][::self.work_memory_sample_rate]
        verb_target_slice = verb_target[work_start: work_end][::self.work_memory_sample_rate]
        noun_target_slice = noun_target[work_start: work_end][::self.work_memory_sample_rate]

        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = work_start - self.long_memory_length, work_start - 1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get future
        if self.future_num_samples > 0:
            future_start, future_end = min(work_end, total_target.shape[0] - 1), \
                min(work_end + self.future_num_samples - 1, total_target.shape[0] - 1)
            future_indices = self.future_sampler(future_start, future_end, self.future_num_samples,
                                                 self.future_sample_rate)
            future_target = total_target[future_indices]
        else:
            future_target = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target_tensor = torch.as_tensor(target_slice.astype(np.float32))

        if self.cfg.MODEL.LSTR.V_N_CLASSIFIER:
            verb_target_tensor = torch.as_tensor(verb_target_slice.astype(np.float32))
            noun_target_tensor = torch.as_tensor(noun_target_slice.astype(np.float32))
            target_tensor = (target_tensor, noun_target_tensor, verb_target_tensor)

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))

            if future_target is not None:
                future_target_tensor = torch.as_tensor(future_target.astype(np.float32))
                result = (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask,
                          (target_tensor, future_target_tensor), session, work_indices, num_frames)
            else:
                result = (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target_tensor, session,
                          work_indices, num_frames)
        else:
            if future_target is not None:
                future_target_tensor = torch.as_tensor(future_target.astype(np.float32))
                result = (fusion_visual_inputs, fusion_motion_inputs, (target_tensor, future_target_tensor), session,
                          work_indices, num_frames)
            else:
                result = (fusion_visual_inputs, fusion_motion_inputs, target_tensor, session, work_indices, num_frames)

        # 【可选优化】：显式删除大变量，帮助 GC 更快回收
        del visual_inputs, motion_inputs, target, verb_target, noun_target, total_target

        return result

    def __len__(self):
        return len(self.inputs)


class _LegacyLSTRDetShortDataLayer(data.Dataset):
    # short feature, (short_ant_gt, past_gt)

    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.data_name = cfg.DATA.DATA_NAME
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        # Anticipation choice
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        # Future choice
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES

        self.training = phase == 'train'
        self.clip_mixup_rate = cfg.DATA.CLIP_MIXUP_RATE
        self.clip_mixup_sample = cfg.DATA.CLIP_MIXUP_SAMPLE
        self.path_to_data = cfg.DATA.EK_EXT_PATH
        self.plus_ratio = 0.25
        self.rgb = dict()
        self.flow = dict()
        self.noun = dict()
        self.target = dict()
        self.verb = dict()

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        segment_list = pd.read_csv(osp.join(self.path_to_data, 'training.csv' if self.training else 'validation.csv'),
                                   names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                   skipinitialspace=True)
        segment_list['len_f'] = segment_list['end_f'] - segment_list['start_f']

        self.segment_by_cls = {cls: segment_list[segment_list['action'] == cls]
                               for cls in range(0, self.cfg.DATA.NUM_CLASSES - 1)}
        self.inputs = []
        for session in self.sessions:
            paths = {
                'target': osp.join(self.data_root, self.target_perframe, session + '.npy'),
                'verb': osp.join(self.data_root, self.target_perframe.replace('target', 'verb'), session + '.npy'),
                'noun': osp.join(self.data_root, self.target_perframe.replace('target', 'noun'), session + '.npy'),
                'rgb': osp.join(self.data_root, self.visual_feature, session + '.npy'),
                'flow': osp.join(self.data_root, self.motion_feature, session + '.npy')
            }
            self.rgb[session] = paths['rgb']
            self.flow[session] = paths['flow']
            self.target[session] = paths['target']
            self.noun[session] = paths['noun']
            self.verb[session] = paths['verb']
            temp_target = np.load(paths['target'], mmap_mode='r')
            video_len = temp_target.shape[0]

            segments_per_session = segment_list[segment_list['video'] == session]
            for segment in segments_per_session.iterrows():
                start_tick = int(segment[1]['start_f'] / 30 * self.cfg.DATA.FPS)
                end_tick = int(segment[1]['end_f'] / 30 * self.cfg.DATA.FPS)

                # 0000AAAAAAAAAAAAAA00000 -> AAAAA(mid) or 000AA(start) or AA000(end)
                exceed_offset = min(self.work_memory_num_samples //2, (end_tick - start_tick) //2)

                start_tick += np.random.randint(max((end_tick + exceed_offset - start_tick), 1)) if self.training else (end_tick - start_tick)
                start_tick = min(start_tick, video_len-1 - self.anticipation_length)
                work_end = start_tick
                work_start = work_end - self.work_memory_length
                segments_before_current = segments_per_session[segments_per_session['end_f'] < segment[1]['start_f']]
                segments_before_end = segments_per_session[segments_per_session['end_f'] < segment[1]['end_f']]
                if work_start < 0:
                    continue
                self.inputs.append([
                    session, work_start, work_end,
                    paths['target'], paths['verb'], paths['noun'],
                    segments_before_end,
                    segments_before_current,
                ])

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        if start < 0:
            start = (end + 1) % sample_rate
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target_path, verb_path, noun_path, segments_work, segments_before = self.inputs[
            index]
        visual_inputs = np.load(self.rgb[session], mmap_mode='r')
        motion_inputs = np.load(self.flow[session], mmap_mode='r')

        # Get target
        total_target = np.load(target_path, mmap_mode='r')
        total_noun_target, total_verb_target = np.load(noun_path, mmap_mode='r'), np.load(verb_path, mmap_mode='r')

        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        target = total_target[work_start: work_end + self.anticipation_length]
        noun_target = total_noun_target[work_start: work_end + self.anticipation_length]
        verb_target = total_verb_target[work_start: work_end + self.anticipation_length]
        target = np.concatenate((target[:self.work_memory_length:self.work_memory_sample_rate],
                                 target[self.work_memory_length::self.anticipation_sample_rate]), axis=0)
        noun_target = np.concatenate((noun_target[:self.work_memory_length:self.work_memory_sample_rate],
                                      noun_target[self.work_memory_length::self.anticipation_sample_rate]), axis=0)
        verb_target = np.concatenate((verb_target[:self.work_memory_length:self.work_memory_sample_rate],
                                      verb_target[self.work_memory_length::self.anticipation_sample_rate]), axis=0)

        # Mixclip++
        if self.training and self.clip_mixup_rate > 0:
            assert segments_work is not None
            segments_work = segments_work[segments_work['end_f'] / 30 * self.cfg.DATA.FPS > work_start]
            prob = None
            num_clip = int(len(segments_work) * 0.5)
            segments_to_mix = segments_work.sample(num_clip, replace=False, weights=prob)
            for segment in segments_to_mix.iterrows():
                start, end = int(segment[1]['start_f'] / 30 * self.cfg.DATA.FPS), int(
                    np.ceil(segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                random_cls = np.random.randint(0, self.cfg.DATA.NUM_CLASSES - 1)
                vid = segment[1]['video']
                # random_cls = segment[1]['action']
                assert vid == session
                segment_random_cls = self.segment_by_cls[random_cls]
                segment_random_cls = segment_random_cls[segment_random_cls['video'] != vid]
                valid_segments = segment_random_cls[segment_random_cls['len_f'] > segment[1]['len_f']]
                if len(valid_segments) == 0:
                    continue
                else:
                    sample_segment = valid_segments.sample(1)
                    new_vid = sample_segment['video'].values[0]
                    new_start_tick = int((sample_segment['start_f'] / 30 * self.cfg.DATA.FPS).iloc[0])
                    new_end_tick = int((sample_segment['end_f'] / 30 * self.cfg.DATA.FPS).iloc[0])

                    new_visual_inputs = np.load(self.rgb[new_vid], mmap_mode='r')
                    new_motion_inputs = np.load(self.flow[new_vid], mmap_mode='r')
                    # 如果需要 target
                    new_target = np.load(self.target[new_vid], mmap_mode='r')
                    new_noun = np.load(self.noun[new_vid], mmap_mode='r')
                    new_verb = np.load(self.verb[new_vid], mmap_mode='r')

                    sel_indices = np.where((work_indices >= start) & (work_indices <= end))
                    shift = np.random.randint(new_end_tick - new_start_tick - len(sel_indices) + 1)

                    work_visual_inputs[sel_indices] = self.plus_ratio * new_visual_inputs[new_start_tick + shift:
                                                                                          new_start_tick + shift + len(
                                                                                              sel_indices)] + (
                                                              1 - self.plus_ratio) * work_visual_inputs[sel_indices]
                    work_motion_inputs[sel_indices] = self.plus_ratio * new_motion_inputs[new_start_tick + shift:
                                                                                          new_start_tick + shift + len(
                                                                                              sel_indices)] + (
                                                              1 - self.plus_ratio) * work_motion_inputs[sel_indices]
                    target[sel_indices] = self.plus_ratio * new_target[new_start_tick + shift: new_start_tick + shift +
                                                                                               len(sel_indices)] + (
                                                  1 - self.plus_ratio) * target[sel_indices]
                    verb_target[sel_indices] = self.plus_ratio * new_verb[
                                                                 new_start_tick + shift:new_start_tick + shift + len(
                                                                     sel_indices)] + (1 - self.plus_ratio) * \
                                               verb_target[sel_indices]
                    noun_target[sel_indices] = self.plus_ratio * new_noun[
                                                                 new_start_tick + shift:new_start_tick + shift + len(
                                                                     sel_indices)] + (1 - self.plus_ratio) * \
                                               noun_target[sel_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            # long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            long_start, long_end = work_start - self.long_memory_length, work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')

            # Clip-level mixup
            if self.training and self.clip_mixup_rate > 0:
                assert segments_before is not None
                segments_before = segments_before[segments_before['end_f'] / 30 * self.cfg.DATA.FPS > long_start]
                if self.clip_mixup_sample == 'uniform':
                    prob = None
                elif self.clip_mixup_sample == 'by_length':
                    prob = (segments_before['len_f']).tolist()
                    prob = prob / np.sum(prob)
                else:
                    raise ValueError
                num_clip = int(len(segments_before) * self.clip_mixup_rate)
                segments_to_mixup = segments_before.sample(num_clip, replace=False, weights=prob)
                for old_segment in segments_to_mixup.iterrows():
                    old_start_tick = int(old_segment[1]['start_f'] / 30 * self.cfg.DATA.FPS)
                    old_end_tick = int(np.ceil(old_segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                    old_action = old_segment[1]['action']
                    old_vid = old_segment[1]['video']
                    assert old_vid == session
                    segment_same_class = self.segment_by_cls[old_action]
                    segment_same_class = segment_same_class[segment_same_class['video'] != old_vid]
                    valid_segments = segment_same_class[segment_same_class['len_f'] > old_segment[1]['len_f']]
                    if len(valid_segments) == 0:
                        continue
                    else:
                        sample_segment = valid_segments.sample(1)
                        new_vid = sample_segment['video'].values[0]
                        new_start_tick = int((sample_segment['start_f'] / 30 * self.cfg.DATA.FPS).iloc[0])
                        new_end_tick = int((sample_segment['end_f'] / 30 * self.cfg.DATA.FPS).iloc[0])

                        new_visual_inputs = np.load(self.rgb[new_vid], mmap_mode='r')
                        new_motion_inputs = np.load(self.flow[new_vid], mmap_mode='r')

                        sel_indices = np.where((long_indices >= old_start_tick) & (long_indices <= old_end_tick))

                        shift = np.random.randint(max(new_end_tick - new_start_tick - len(sel_indices) + 1, 1))
                        long_visual_inputs[sel_indices] = new_visual_inputs[
                                                          new_start_tick + shift:new_start_tick + shift + len(
                                                              sel_indices)]
                        long_motion_inputs[sel_indices] = new_motion_inputs[
                                                          new_start_tick + shift:new_start_tick + shift + len(
                                                              sel_indices)]

        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get future
        if self.future_num_samples > 0:
            future_start, future_end = min(work_start, total_target.shape[0] - 1), min(work_start + self.future_num_samples - 1, total_target.shape[0] - 1)
            future_indices = self.future_sampler(future_start, future_end, self.future_num_samples, self.future_sample_rate)
            future_target = total_target[future_indices]
            future_noun_target = total_noun_target[future_indices]
            future_verb_target = total_verb_target[future_indices]
        else:
            future_target = None
            future_noun_target = None
            future_verb_target = None

        # Get all memory
        if (long_visual_inputs is not None and long_motion_inputs is not None):
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))
        verb_target = torch.as_tensor(verb_target.astype(np.float32))
        noun_target = torch.as_tensor(noun_target.astype(np.float32))

        if future_target is not None:
            future_noun_target = torch.as_tensor(future_noun_target.astype(np.float32))
            future_verb_target = torch.as_tensor(future_verb_target.astype(np.float32))
            target = (target, noun_target, verb_target, future_target, future_noun_target, future_verb_target)
        else:
            target = (target, verb_target, noun_target)

        assert fusion_visual_inputs.ndim == fusion_motion_inputs.ndim

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, fusion_motion_inputs, target)

    def __len__(self):
        return len(self.inputs)


class _EK100ShortMemoryMixin:
    _SEGMENT_COLUMNS = ['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action']

    def _load_segment_metadata(self):
        split = 'training.csv' if self.training else 'validation.csv'
        self.segment_list = pd.read_csv(
            osp.join(self.path_to_data, split),
            names=self._SEGMENT_COLUMNS,
            skipinitialspace=True,
        ).reset_index(drop=True)
        self.segment_list['len_f'] = (
            self.segment_list['end_f'] - self.segment_list['start_f']
        )
        self.segment_indices_by_session = {
            str(session): np.asarray(indices, dtype=np.int32)
            for session, indices in self.segment_list.groupby('video', sort=False).groups.items()
        }
        self.segment_indices_by_cls = {
            int(action): np.asarray(indices, dtype=np.int32)
            for action, indices in self.segment_list.groupby('action', sort=False).groups.items()
        }

    def _register_session_paths(self, session):
        if session not in self.rgb:
            self.rgb[session] = osp.join(self.data_root, self.visual_feature, session + '.npy')
            self.flow[session] = osp.join(self.data_root, self.motion_feature, session + '.npy')
            self.target[session] = osp.join(self.data_root, self.target_perframe, session + '.npy')
            self.verb[session] = osp.join(
                self.data_root,
                self.target_perframe.replace('target', 'verb'),
                session + '.npy',
            )
            self.noun[session] = osp.join(
                self.data_root,
                self.target_perframe.replace('target', 'noun'),
                session + '.npy',
            )

    def _segments_for_session(self, session):
        indices = self.segment_indices_by_session.get(str(session))
        if indices is None:
            return self.segment_list.iloc[0:0]
        return self.segment_list.iloc[indices]

    def _segments_for_action(self, action):
        indices = self.segment_indices_by_cls.get(int(action))
        if indices is None:
            return self.segment_list.iloc[0:0]
        return self.segment_list.iloc[indices]

    def _mixup_segments(self, session, segment_index):
        segment = self.segment_list.iloc[int(segment_index)]
        segments = self._segments_for_session(session)
        segments_work = segments[segments['end_f'] < segment['end_f']]
        segments_before = segments[segments['end_f'] < segment['start_f']]
        return segments_work, segments_before


class LSTRDetShortDataLayer(_EK100ShortMemoryMixin, data.Dataset):
    # short feature, (short_ant_gt, past_gt)

    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.data_name = cfg.DATA.DATA_NAME
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        # Anticipation choice
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        # Future choice
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES

        self.training = phase == 'train'
        self.clip_mixup_rate = cfg.DATA.CLIP_MIXUP_RATE
        self.clip_mixup_sample = cfg.DATA.CLIP_MIXUP_SAMPLE
        self.path_to_data = cfg.DATA.EK_EXT_PATH
        self.plus_ratio = 0.25
        self.rgb = dict()
        self.flow = dict()
        self.noun = dict()
        self.target = dict()
        self.verb = dict()

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        if not hasattr(self, 'segment_list'):
            self._load_segment_metadata()
        self.inputs = []
        for session in self.sessions:
            self._register_session_paths(session)
            temp_target = np.load(self.target[session], mmap_mode='r')
            video_len = temp_target.shape[0]
            del temp_target

            segment_indices = self.segment_indices_by_session.get(str(session), ())
            for segment_index in segment_indices:
                segment = self.segment_list.iloc[int(segment_index)]
                start_tick = int(segment['start_f'] / 30 * self.cfg.DATA.FPS)
                end_tick = int(segment['end_f'] / 30 * self.cfg.DATA.FPS)

                # 0000AAAAAAAAAAAAAA00000 -> AAAAA(mid) or 000AA(start) or AA000(end)
                exceed_offset = min(self.work_memory_num_samples //2, (end_tick - start_tick) //2)

                start_tick += np.random.randint(max((end_tick + exceed_offset - start_tick), 1)) if self.training else (end_tick - start_tick)
                start_tick = min(start_tick, video_len-1 - self.anticipation_length)
                work_end = start_tick
                work_start = work_end - self.work_memory_length
                if work_start < 0:
                    continue
                self.inputs.append((session, work_start, work_end, int(segment_index)))

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        if start < 0:
            start = (end + 1) % sample_rate
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, segment_index = self.inputs[index]
        visual_inputs = np.load(self.rgb[session], mmap_mode='r')
        motion_inputs = np.load(self.flow[session], mmap_mode='r')

        # Get target
        total_target = np.load(self.target[session], mmap_mode='r')
        total_noun_target = np.load(self.noun[session], mmap_mode='r')
        total_verb_target = np.load(self.verb[session], mmap_mode='r')

        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        target = total_target[work_start: work_end + self.anticipation_length]
        noun_target = total_noun_target[work_start: work_end + self.anticipation_length]
        verb_target = total_verb_target[work_start: work_end + self.anticipation_length]
        target = np.concatenate((target[:self.work_memory_length:self.work_memory_sample_rate],
                                 target[self.work_memory_length::self.anticipation_sample_rate]), axis=0)
        noun_target = np.concatenate((noun_target[:self.work_memory_length:self.work_memory_sample_rate],
                                      noun_target[self.work_memory_length::self.anticipation_sample_rate]), axis=0)
        verb_target = np.concatenate((verb_target[:self.work_memory_length:self.work_memory_sample_rate],
                                      verb_target[self.work_memory_length::self.anticipation_sample_rate]), axis=0)

        # Mixclip++
        if self.training and self.clip_mixup_rate > 0:
            segments_work, segments_before = self._mixup_segments(session, segment_index)
            segments_work = segments_work[segments_work['end_f'] / 30 * self.cfg.DATA.FPS > work_start]
            prob = None
            num_clip = int(len(segments_work) * 0.5)
            segments_to_mix = segments_work.sample(num_clip, replace=False, weights=prob)
            for segment in segments_to_mix.iterrows():
                start, end = int(segment[1]['start_f'] / 30 * self.cfg.DATA.FPS), int(
                    np.ceil(segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                random_cls = np.random.randint(0, self.cfg.DATA.NUM_CLASSES - 1)
                vid = segment[1]['video']
                # random_cls = segment[1]['action']
                assert vid == session
                segment_random_cls = self._segments_for_action(random_cls)
                segment_random_cls = segment_random_cls[segment_random_cls['video'] != vid]
                valid_segments = segment_random_cls[segment_random_cls['len_f'] > segment[1]['len_f']]
                if len(valid_segments) == 0:
                    continue
                else:
                    sample_segment = valid_segments.sample(1)
                    new_vid = sample_segment['video'].values[0]
                    self._register_session_paths(new_vid)
                    new_start_tick = int((sample_segment['start_f'] / 30 * self.cfg.DATA.FPS).iloc[0])
                    new_end_tick = int((sample_segment['end_f'] / 30 * self.cfg.DATA.FPS).iloc[0])

                    new_visual_inputs = np.load(self.rgb[new_vid], mmap_mode='r')
                    new_motion_inputs = np.load(self.flow[new_vid], mmap_mode='r')
                    # 如果需要 target
                    new_target = np.load(self.target[new_vid], mmap_mode='r')
                    new_noun = np.load(self.noun[new_vid], mmap_mode='r')
                    new_verb = np.load(self.verb[new_vid], mmap_mode='r')

                    sel_indices = np.where((work_indices >= start) & (work_indices <= end))
                    shift = np.random.randint(new_end_tick - new_start_tick - len(sel_indices) + 1)

                    work_visual_inputs[sel_indices] = self.plus_ratio * new_visual_inputs[new_start_tick + shift:
                                                                                          new_start_tick + shift + len(
                                                                                              sel_indices)] + (
                                                              1 - self.plus_ratio) * work_visual_inputs[sel_indices]
                    work_motion_inputs[sel_indices] = self.plus_ratio * new_motion_inputs[new_start_tick + shift:
                                                                                          new_start_tick + shift + len(
                                                                                              sel_indices)] + (
                                                              1 - self.plus_ratio) * work_motion_inputs[sel_indices]
                    target[sel_indices] = self.plus_ratio * new_target[new_start_tick + shift: new_start_tick + shift +
                                                                                               len(sel_indices)] + (
                                                  1 - self.plus_ratio) * target[sel_indices]
                    verb_target[sel_indices] = self.plus_ratio * new_verb[
                                                                 new_start_tick + shift:new_start_tick + shift + len(
                                                                     sel_indices)] + (1 - self.plus_ratio) * \
                                               verb_target[sel_indices]
                    noun_target[sel_indices] = self.plus_ratio * new_noun[
                                                                 new_start_tick + shift:new_start_tick + shift + len(
                                                                     sel_indices)] + (1 - self.plus_ratio) * \
                                               noun_target[sel_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            # long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            long_start, long_end = work_start - self.long_memory_length, work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')

            # Clip-level mixup
            if self.training and self.clip_mixup_rate > 0:
                assert segments_before is not None
                segments_before = segments_before[segments_before['end_f'] / 30 * self.cfg.DATA.FPS > long_start]
                if self.clip_mixup_sample == 'uniform':
                    prob = None
                elif self.clip_mixup_sample == 'by_length':
                    prob = (segments_before['len_f']).tolist()
                    prob = prob / np.sum(prob)
                else:
                    raise ValueError
                num_clip = int(len(segments_before) * self.clip_mixup_rate)
                segments_to_mixup = segments_before.sample(num_clip, replace=False, weights=prob)
                for old_segment in segments_to_mixup.iterrows():
                    old_start_tick = int(old_segment[1]['start_f'] / 30 * self.cfg.DATA.FPS)
                    old_end_tick = int(np.ceil(old_segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                    old_action = old_segment[1]['action']
                    old_vid = old_segment[1]['video']
                    assert old_vid == session
                    segment_same_class = self._segments_for_action(old_action)
                    segment_same_class = segment_same_class[segment_same_class['video'] != old_vid]
                    valid_segments = segment_same_class[segment_same_class['len_f'] > old_segment[1]['len_f']]
                    if len(valid_segments) == 0:
                        continue
                    else:
                        sample_segment = valid_segments.sample(1)
                        new_vid = sample_segment['video'].values[0]
                        self._register_session_paths(new_vid)
                        new_start_tick = int((sample_segment['start_f'] / 30 * self.cfg.DATA.FPS).iloc[0])
                        new_end_tick = int((sample_segment['end_f'] / 30 * self.cfg.DATA.FPS).iloc[0])

                        new_visual_inputs = np.load(self.rgb[new_vid], mmap_mode='r')
                        new_motion_inputs = np.load(self.flow[new_vid], mmap_mode='r')

                        sel_indices = np.where((long_indices >= old_start_tick) & (long_indices <= old_end_tick))

                        shift = np.random.randint(max(new_end_tick - new_start_tick - len(sel_indices) + 1, 1))
                        long_visual_inputs[sel_indices] = new_visual_inputs[
                                                          new_start_tick + shift:new_start_tick + shift + len(
                                                              sel_indices)]
                        long_motion_inputs[sel_indices] = new_motion_inputs[
                                                          new_start_tick + shift:new_start_tick + shift + len(
                                                              sel_indices)]

        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get future
        if self.future_num_samples > 0:
            future_start, future_end = min(work_start, total_target.shape[0] - 1), min(work_start + self.future_num_samples - 1, total_target.shape[0] - 1)
            future_indices = self.future_sampler(future_start, future_end, self.future_num_samples, self.future_sample_rate)
            future_target = total_target[future_indices]
            future_noun_target = total_noun_target[future_indices]
            future_verb_target = total_verb_target[future_indices]
        else:
            future_target = None
            future_noun_target = None
            future_verb_target = None

        # Get all memory
        if (long_visual_inputs is not None and long_motion_inputs is not None):
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))
        verb_target = torch.as_tensor(verb_target.astype(np.float32))
        noun_target = torch.as_tensor(noun_target.astype(np.float32))

        if future_target is not None:
            future_noun_target = torch.as_tensor(future_noun_target.astype(np.float32))
            future_verb_target = torch.as_tensor(future_verb_target.astype(np.float32))
            target = (target, noun_target, verb_target, future_target, future_noun_target, future_verb_target)
        else:
            target = (target, verb_target, noun_target)

        assert fusion_visual_inputs.ndim == fusion_motion_inputs.ndim

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, fusion_motion_inputs, target)

    def __len__(self):
        return len(self.inputs)

class LSTRAntShortDataLayer(_EK100ShortMemoryMixin, data.Dataset):
    # short feature, (short_ant_gt, past_gt)

    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.data_name = cfg.DATA.DATA_NAME
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        # Anticipation choice
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        # Future choice
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES

        self.training = phase == 'train'
        self.clip_mixup_rate = cfg.DATA.CLIP_MIXUP_RATE
        self.clip_mixup_sample = cfg.DATA.CLIP_MIXUP_SAMPLE
        self.path_to_data = cfg.DATA.EK_EXT_PATH
        self.plus_ratio = 0.25
        self.rgb = dict()
        self.flow = dict()
        self.noun = dict()
        self.target = dict()
        self.verb = dict()

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        if not hasattr(self, 'segment_list'):
            self._load_segment_metadata()
        self.inputs = []
        for session in self.sessions:
            self._register_session_paths(session)
            temp_target = np.load(self.target[session], mmap_mode='r')
            video_len = temp_target.shape[0]
            del temp_target

            segment_indices = self.segment_indices_by_session.get(str(session), ())
            for segment_index in segment_indices:
                segment = self.segment_list.iloc[int(segment_index)]
                start_tick = int(segment['start_f'] / 30 * self.cfg.DATA.FPS)
                end_tick = int(segment['end_f'] / 30 * self.cfg.DATA.FPS)

                start_tick += np.random.randint(max(1, self.anticipation_length)) if self.training else 0
                start_tick = min(start_tick, end_tick)
                work_end = start_tick - self.anticipation_length
                work_start = work_end - self.work_memory_length
                if work_start < 0:
                    continue
                self.inputs.append((session, work_start, work_end, int(segment_index)))

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        if start < 0:
            start = (end + 1) % sample_rate
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, segment_index = self.inputs[index]
        visual_inputs = np.load(self.rgb[session], mmap_mode='r')
        motion_inputs = np.load(self.flow[session], mmap_mode='r')

        # Get target
        total_target = np.load(self.target[session], mmap_mode='r')
        total_noun_target = np.load(self.noun[session], mmap_mode='r')
        total_verb_target = np.load(self.verb[session], mmap_mode='r')

        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        target = total_target[work_start: work_end + self.anticipation_length]
        noun_target = total_noun_target[work_start: work_end + self.anticipation_length]
        verb_target = total_verb_target[work_start: work_end + self.anticipation_length]
        target = np.concatenate((target[:self.work_memory_length:self.work_memory_sample_rate],
                                 target[self.work_memory_length::self.anticipation_sample_rate]), axis=0)
        noun_target = np.concatenate((noun_target[:self.work_memory_length:self.work_memory_sample_rate],
                                      noun_target[self.work_memory_length::self.anticipation_sample_rate]), axis=0)
        verb_target = np.concatenate((verb_target[:self.work_memory_length:self.work_memory_sample_rate],
                                      verb_target[self.work_memory_length::self.anticipation_sample_rate]), axis=0)

        # Mixclip++
        if self.training and self.clip_mixup_rate > 0:
            segments_work, segments_before = self._mixup_segments(session, segment_index)
            segments_work = segments_work[segments_work['end_f'] / 30 * self.cfg.DATA.FPS > work_start]
            prob = None
            num_clip = int(len(segments_work) * 0.5)
            segments_to_mix = segments_work.sample(num_clip, replace=False, weights=prob)
            for segment in segments_to_mix.iterrows():
                start, end = int(segment[1]['start_f'] / 30 * self.cfg.DATA.FPS), int(
                    np.ceil(segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                random_cls = np.random.randint(0, self.cfg.DATA.NUM_CLASSES - 1)
                vid = segment[1]['video']
                # random_cls = segment[1]['action']
                assert vid == session
                segment_random_cls = self._segments_for_action(random_cls)
                segment_random_cls = segment_random_cls[segment_random_cls['video'] != vid]
                valid_segments = segment_random_cls[segment_random_cls['len_f'] > segment[1]['len_f']]
                if len(valid_segments) == 0:
                    continue
                else:
                    sample_segment = valid_segments.sample(1)
                    new_vid = sample_segment['video'].values[0]
                    self._register_session_paths(new_vid)
                    new_start_tick = int(sample_segment['start_f'] / 30 * self.cfg.DATA.FPS)
                    new_end_tick = int(sample_segment['end_f'] / 30 * self.cfg.DATA.FPS)

                    new_visual_inputs = np.load(self.rgb[new_vid], mmap_mode='r')
                    new_motion_inputs = np.load(self.flow[new_vid], mmap_mode='r')
                    new_target = np.load(self.target[new_vid], mmap_mode='r')
                    new_noun = np.load(self.noun[new_vid], mmap_mode='r')
                    new_verb = np.load(self.verb[new_vid], mmap_mode='r')

                    sel_indices = np.where((work_indices >= start) & (work_indices <= end))
                    shift = np.random.randint(max(new_end_tick - new_start_tick - len(sel_indices) + 1, 1))

                    work_visual_inputs[sel_indices] = self.plus_ratio * new_visual_inputs[new_start_tick + shift:
                                                        new_start_tick + shift + len(sel_indices)] + \
                                                      (1 - self.plus_ratio) * work_visual_inputs[sel_indices]
                    work_motion_inputs[sel_indices] = self.plus_ratio * new_motion_inputs[new_start_tick + shift:
                                                                new_start_tick + shift + len(sel_indices)] + \
                                                      (1 - self.plus_ratio) * work_motion_inputs[sel_indices]
                    target[sel_indices] = self.plus_ratio * new_target[new_start_tick + shift: new_start_tick + shift
                                                      + len(sel_indices)] + (1 - self.plus_ratio) * target[sel_indices]
                    verb_target[sel_indices] = self.plus_ratio * new_verb[new_start_tick + shift:new_start_tick + shift
                                                      + len(sel_indices)] + (1 - self.plus_ratio) * verb_target[sel_indices]
                    noun_target[sel_indices] = self.plus_ratio * new_noun[new_start_tick + shift:new_start_tick + shift
                                                      + len(sel_indices)] + (1 - self.plus_ratio) * noun_target[sel_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = work_start - self.long_memory_length, work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')

            # Clip-level mixup
            if self.training and self.clip_mixup_rate > 0:
                segments_before = segments_before[segments_before['end_f'] / 30 * self.cfg.DATA.FPS > long_start]
                if self.clip_mixup_sample == 'uniform':
                    prob = None
                elif self.clip_mixup_sample == 'by_length':
                    prob = (segments_before['len_f']).tolist()
                    prob = prob / np.sum(prob)
                else:
                    raise ValueError
                num_clip = int(len(segments_before) * self.clip_mixup_rate)
                segments_to_mixup = segments_before.sample(num_clip, replace=False, weights=prob)
                for old_segment in segments_to_mixup.iterrows():
                    old_start_tick = int(old_segment[1]['start_f'] / 30 * self.cfg.DATA.FPS)
                    old_end_tick = int(np.ceil(old_segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                    old_action = old_segment[1]['action']
                    old_vid = old_segment[1]['video']
                    assert old_vid == session
                    segment_same_class = self._segments_for_action(old_action)
                    segment_same_class = segment_same_class[segment_same_class['video'] != old_vid]
                    valid_segments = segment_same_class[segment_same_class['len_f'] > old_segment[1]['len_f']]
                    if len(valid_segments) == 0:
                        continue
                    else:
                        sample_segment = valid_segments.sample(1)
                        new_vid = sample_segment['video'].values[0]
                        self._register_session_paths(new_vid)
                        new_start_tick = int(sample_segment['start_f'] / 30 * self.cfg.DATA.FPS)
                        new_end_tick = int(sample_segment['end_f'] / 30 * self.cfg.DATA.FPS)

                        new_visual_inputs = np.load(self.rgb[new_vid], mmap_mode='r')
                        new_motion_inputs = np.load(self.flow[new_vid], mmap_mode='r')

                        sel_indices = np.where((long_indices >= old_start_tick) & (long_indices <= old_end_tick))

                        shift = np.random.randint(max(new_end_tick - new_start_tick - len(sel_indices) + 1, 1))
                        long_visual_inputs[sel_indices] = new_visual_inputs[
                                                          new_start_tick + shift:new_start_tick + shift + len(sel_indices)]
                        long_motion_inputs[sel_indices] = new_motion_inputs[
                                                          new_start_tick + shift:new_start_tick + shift + len(sel_indices)]

        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get future
        if self.future_num_samples > 0:
            future_start, future_end = min(work_start, total_target.shape[0] - 1), min(work_start + self.future_num_samples - 1, total_target.shape[0] - 1)
            future_indices = self.future_sampler(future_start, future_end, self.future_num_samples, self.future_sample_rate)
            future_target = total_target[future_indices]
            future_noun_target = total_noun_target[future_indices]
            future_verb_target = total_verb_target[future_indices]
        else:
            future_target = None
            future_noun_target = None
            future_verb_target = None

        # Get all memory
        if (long_visual_inputs is not None and long_motion_inputs is not None):
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))
        verb_target = torch.as_tensor(verb_target.astype(np.float32))
        noun_target = torch.as_tensor(noun_target.astype(np.float32))

        if future_target is not None:
            future_noun_target = torch.as_tensor(future_noun_target.astype(np.float32))
            future_verb_target = torch.as_tensor(future_verb_target.astype(np.float32))
            target = (target, noun_target, verb_target, future_target, future_noun_target, future_verb_target)
        else:
            target = (target, verb_target, noun_target)

        assert fusion_visual_inputs.ndim == fusion_motion_inputs.ndim

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, fusion_motion_inputs, target)

    def __len__(self):
        return len(self.inputs)
