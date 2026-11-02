# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    'build_dataset',
    'build_data_loader',
]

import torch.utils.data as data


from rekognition_online_action_detection.utils.registry import Registry

DATA_LAYERS = Registry()


def build_dataset(cfg, phase, tag=''):
    data_layer = DATA_LAYERS[tag + cfg.DATA.DATA_NAME] #cfg.MODEL.MODEL_NAME
    return data_layer(cfg, phase)


def build_data_loader(cfg, phase):
    if cfg.DATA.DATA_NAME == 'EK100':
        from rekognition_online_action_detection.datasets.perframe_data_layers import LSTRDetShortDataLayer, \
            LSTRAntShortDataLayer
        loader_kwargs = {
            'batch_size': cfg.DATA_LOADER.BATCH_SIZE,
            'shuffle': phase == 'train',
            'num_workers': cfg.DATA_LOADER.NUM_WORKERS,
            'pin_memory': cfg.DATA_LOADER.PIN_MEMORY,
        }
        if cfg.DATA_LOADER.NUM_WORKERS > 0:
            # EK100 batches are large; the default prefetch factor of two can
            # double each worker's queued host-memory footprint.
            loader_kwargs['prefetch_factor'] = 1
        if cfg.DATA_LOADER.TYPE == 'det':
            data_loader = data.DataLoader(
                dataset=LSTRDetShortDataLayer(cfg, phase),
                **loader_kwargs,
            )
        elif cfg.DATA_LOADER.TYPE == 'ant':
            data_loader = data.DataLoader(
                dataset=LSTRAntShortDataLayer(cfg, phase),
                **loader_kwargs,
            )
        else:
            exit(2)
    else:
        data_loader = data.DataLoader(
            dataset=build_dataset(cfg, phase, cfg.DATA_LOADER.TAG),
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            shuffle=True if phase == 'train' else False,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        )
    return data_loader
