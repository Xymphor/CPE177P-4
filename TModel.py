from easydict import EasyDict as edict
import os
import numpy as np
import matplotlib.pyplot as plt
import mindspore
import mindspore.dataset as ds
from mindspore.dataset.vision import c_transforms as vision
from mindspore import context
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import Tensor
from mindspore.train.serialization import export
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.ops as ops

context.set_context(mode =context.GRAPH_MODE, device_target = "CPU")

cfg = edict({
    'data_path':'Mushroom/Mushroom_photos_train',
    'test_path':'Mushroom/Mushroom_photos_test',
    'data_size': 2500,
    'HEIGHT': 224,
    'WIDTH': 224,
    '_R_MEAN': 123.68,
    '_G_MEAN': 116.78,
    '_B_MEAN': 103.94,
    '_R_STD': 1,
    '_G_STD': 1,
    '_B_STD': 1,
    '_RESIZE_SIDE_MIN':256,
    '_RESIZE_SIDE_MAX':512,

    'batch_size': 32,
    'num_class': 5,
    'epoch_size': 10,
    'loss_scale_num': 1024,

    'prefix': 'resnet-ai',
    'directory': './mode_resnet',
    'save_checkpoint_steps': 10,
    })

def read_data(path,config,usage="train"):
    dataset = ds.ImageFolderDataset(path, class_indexing = {'Amanita':0, 'Boletus':1, 'Cortinarius':2, 'Lactarius':3, 'Russula':4})
    decode_op = vision.Decode()
    normalize_op = vision.Normalize(mean=[cfg._R_MEAN,cfg._G_MEAN,cfg._B_MEAN], std=[cfg._R_STD,cfg._G_STD,cfg._B_STD])
    resize_op = vision.Resize(cfg._RESIZE_SIDE_MIN)
    center_crop_op = vision.CenterCrop((cfg.HEIGHT, cfg.WIDTH))
    horizontal_flip_op = vision.RandomHorizontalFlip()
    channelswap_op = vision.HWC2CHW()
    random_crop_decode_resize_op = vision.RandomCropDecodeResize((cfg.HEIGHT, cfg.WIDTH), (0.5,1.0), (1.0,1.0),max_attempts=100)

    if usage == 'train':
        dataset = dataset.map(input_columns="image", operations=random_crop_decode_resize_op)
        dataset = dataset.map(input_columns="image", operations=horizontal_flip_op)
    else:
        dataset = dataset.map(input_columns="image", operations=decode_op)
        dataset = dataset.map(input_columns="image", operations=resize_op)
        dataset = dataset.map(input_columns="image", operations=center_crop_op)
    
    dataset = dataset.map(input_columns="image", operations=normalize_op)
    dataset = dataset.map(input_columns="image", operations=channelswap_op)

    if usage == 'train':
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(cfg.batch_size, drop_remainder=True)
    else:
        dataset=dataset.batch(1,drop_remainder=True)
    
    dataset=dataset.repeat(1)

    dataset.map_model = 4
    
    return dataset

de_train = read_data(cfg.data_path,cfg,usage="train")
de_test = read_data(cfg.test_path,cfg,usage="test")
print('Number of training datasets: ', de_train.get_dataset_size()*cfg.batch_size)
print('Label style of an image: ',de_test.get_dataset_size())

data_next = de_train.create_dict_iterator(output_numpy=True).__next__()
print('Number of channels/Image length/width: ', data_next['image'][0,...].shape)
print('Label style of an image: ',data_next['label'][0])

plt.figure()
plt.imshow(data_next['image'][0,0,...])
plt.colorbar()
plt.grid(False)
plt.show()


