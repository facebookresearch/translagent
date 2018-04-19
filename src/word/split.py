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

from util import scr_path

random = np.random

def total_len(data):
    return sum( [ len(v) for k,v in data.iteritems() ] )

# Splits 1/k of given images into validation images, leaves rest as training images.
def split_data_by_k(data, k, dic):
    if dic:
        data1, data2 = OrderedDict(), OrderedDict()

        for k1, v1 in data.iteritems():
            assert len(v1) >= k
            mid = len(v1) / k
            data1[k1] = v1[mid:]
            data2[k1] = v1[:mid]

        print "{} total images".format( sum( [ len(data[key]) for key in data.keys() ] ) )
        print "{} (1) images".format( sum( [ len(data1[key]) for key in data1.keys() ] ) )
        print "{} (2) images".format( sum( [ len(data2[key]) for key in data2.keys() ] ) )

    else:
        data1, data2 = [], []

        for v1 in data:
            assert len(v1) >= k
            mid = len(v1) / k
            data1.append( v1[mid:] )
            data2.append( v1[:mid] )

        print "{} total images".format( sum( [ len(d) for d in data ] ) )
        print "{} (1) images".format( sum( [ len(d) for d in data1 ] ) )
        print "{} (2) images".format( sum( [ len(d) for d in data2 ] ) )

    return data1, data2

def split_bergsma(l1, l2):
    l1_data = torch.load("{}/data/word/{}_2048.pt".format(scr_path(), l1))
    l2_data = torch.load("{}/data/word/{}_2048.pt".format(scr_path(), l2))

    keys = l1_data.keys()
    keys = np.array([x for x in keys if x in l2_data.keys()])
    print "before : {} keys".format(len(keys))
    keys = [k for k in keys if len(l1_data[k]) >= 5 and len(l2_data[k]) >= 5 ]
    print "after : {} keys".format(len(keys))
    original_keys = keys

    l1_data_, l2_data_ = {}, {}
    for idx, k in enumerate(keys):
        l1_data_[idx] = l1_data[k]
        l2_data_[idx] = l2_data[k]
    # list X list X np.array
    # 440 X 15-20 X 300

    l1_train, l1_valid = split_data_by_k(l1_data_, 5, True)
    l2_train, l2_valid = split_data_by_k(l2_data_, 5, True)

    print "{} {} keys {} images".format(l1.upper(), len(l1_train), sum([len(x) for x in l1_train.values()]) )
    print "{} {} keys {} images".format(l2.upper(), len(l2_train), sum([len(x) for x in l2_train.values()]) )
    keys = np.arange(len(l1_data_))

    return l1_train, l1_valid, l2_train, l2_valid, keys, original_keys

