# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_criterion']

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os.path as osp
from functools import partial

from rekognition_online_action_detection.utils.registry import Registry

CRITERIONS = Registry()


@CRITERIONS.register('MCE')
class MultipCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(MultipCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), dim=1)
            if (target[:, self.ignore_index] != 1).sum().item() == 0:
                return torch.mean(torch.zeros_like(output))
            if self.reduction == 'mean':
                return torch.mean(output[target[:, self.ignore_index] != 1])
            elif self.reduction == 'sum':
                return torch.sum(output[target[:, self.ignore_index] != 1])
            else:
                return output[target[:, self.ignore_index] != 1]
        else:
            output = torch.sum(-target * logsoftmax(input), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output)
            elif self.reduction == 'sum':
                return torch.sum(output)
            else:
                return output


@CRITERIONS.register('MCE_EQL')
class MultipCrossEntropyEqualizedLoss(nn.Module):

    def __init__(self, gamma=0.95, lambda_=1.76e-3, reduction='mean', ignore_index=-100,
                 anno_path='external/rulstm/RULSTM/data/ek55/'):
        super(MultipCrossEntropyEqualizedLoss, self).__init__()

        # get label distribution
        segment_list = pd.read_csv(osp.join(anno_path, 'training.csv'),
                                   names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                   skipinitialspace=True)
        freq_info = np.zeros((max(segment_list['action']) + 1,))
        assert ignore_index == 0
        for segment in segment_list.iterrows():
            freq_info[segment[1]['action']] += 1.
        freq_info = freq_info / freq_info.sum()
        self.freq_info = torch.FloatTensor(freq_info)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        bg_target = target[:, self.ignore_index]
        notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
        input = input[:, notice_index]
        target = target[:, notice_index]

        weight = input.new_zeros(len(notice_index))
        weight[self.freq_info < self.lambda_] = 1.
        weight = weight.view(1, -1).repeat(input.shape[0], 1)

        eql_w = 1 - (torch.rand_like(target) < self.gamma) * weight * (1 - target)
        input = torch.log(eql_w + 1e-8) + input

        output = torch.sum(-target * logsoftmax(input), dim=1)
        if (bg_target != 1).sum().item() == 0:
            return torch.mean(torch.zeros_like(output))
        if self.reduction == 'mean':
            return torch.mean(output[bg_target != 1])
        elif self.reduction == 'sum':
            return torch.sum(output[bg_target != 1])
        else:
            return output[bg_target != 1]

def get_marginal_indexes(actions, mode):
    """For each verb/noun retrieve the list of actions containing that verb/name
        Input:
            mode: "verb" or "noun"
        Output:
            a list of numpy array of indexes. If verb/noun 3 is contained in actions 2,8,19,
            then output[3] will be np.array([2,8,19])
    """
    vi = []
    for v in range(actions[mode].max()+1):
        vals = actions[actions[mode] == v].index.values
        if len(vals) > 0:
            vi.append(vals)
        else:
            vi.append(np.array([0]))
    return vi
@CRITERIONS.register('MARGINAL')
class MarginalCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(MarginalCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index
        actions = pd.read_csv(osp.join('external/rulstm/RULSTM/data/ek100/', 'actions.csv'), index_col='id')
        self.vi = get_marginal_indexes(actions, 'verb')
        self.ni = get_marginal_indexes(actions, 'noun')

    def forward(self, action_scores, target, dtype = 'noun'):
        assert dtype in ['noun', 'verb']
        action_scores = action_scores[:, 1:] # remove background
        action_probs = F.softmax(action_scores, dim = -1)

        def marginalize_torch(probs, indexes):
            mprobs = []
            for ilist in indexes:
                ilist = list(ilist)
                mprobs.append((probs[:, ilist].sum(1)).view(-1, 1))
            mprobs = torch.cat(mprobs, dim=-1)
            return mprobs

        if dtype == 'noun':
            scores = marginalize_torch(action_probs, self.ni)
        else:
            scores = marginalize_torch(action_probs, self.vi)

        new_target = target[:, 1:]
        valid_index = target[:, 0] != 1
        output = torch.sum(-new_target * torch.log(scores), dim=1)
        return torch.mean(output[valid_index])

def build_criterion(cfg, device=None):
    criterion = {}
    for name, params in cfg.MODEL.CRITERIONS:
        if name in CRITERIONS:
            #params['cfg'] = cfg
            if 'ignore_index' not in params:
                params['ignore_index'] = cfg.DATA.IGNORE_INDEX
            criterion[name] = CRITERIONS[name](**params).to(device)
        else:
            raise RuntimeError('Unknown criterion: {}'.format(name))
    return criterion
