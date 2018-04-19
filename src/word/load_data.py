# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import cPickle as pkl
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch.utils.serialization import load_lua

from util import *

random = np.random

def next_batch_dist_batch(data, batch_size, num_dist, num_cat):
    lsn_imgs, spk_imgs, labels, whichs = [], [], [], []
    keys = data.keys()
    assert len(keys) >= num_dist

    for batch_idx in xrange(batch_size):
        rand_labels = random.choice( keys, num_dist, replace=False ) # (num_dist)

        num_imgs = [len(data[label]) for label in rand_labels] # (num_dist)
        img_indices = [random.randint(0, num_img) for num_img in num_imgs]
        all_images = [ data[label][img_idx] for label, img_idx in zip(rand_labels, img_indices) ] # (num_dist, 2048)

        which = random.randint(0, num_dist) # (1)
        label_ = rand_labels[which] # (1)

        lsn_imgs.append(all_images)  # (batch_size, num_dist, 2048)
        spk_imgs.append(all_images[which]) # (batch_size, 2048)
        labels.append(label_) # (batch_size)
        whichs.append(which) # (batch_size)

    lsn_imgs, spk_imgs, labels, whichs = np.array(lsn_imgs), np.array(spk_imgs), np.array(labels), np.array(whichs)

    spk_imgs = Variable(torch.stack(spk_imgs), requires_grad=False).cuda().view(batch_size, -1)
    lsn_imgs = [torch.stack(x) for x in lsn_imgs]
    lsn_imgs = Variable(torch.stack(lsn_imgs), requires_grad=False).cuda().view(batch_size, num_dist, -1)
    labels_ = Variable(torch.LongTensor(labels), requires_grad=False).cuda().view(batch_size)
    whichs = Variable(torch.LongTensor(whichs), requires_grad=False).cuda().view(batch_size)
    label_onehot = Variable(idx_to_onehot(labels, num_cat), requires_grad=False).cuda().view(batch_size, -1)

    return (spk_imgs, lsn_imgs, label_onehot, labels_, whichs)

