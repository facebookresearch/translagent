# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import OrderedDict
import numpy as np

import torch
from torch.autograd import Variable

from split import split_data_by_k
from util import scr_path

# input : (500 concepts X 2048 dimension)
def compute_dot_same_shape(data1, data2):
    assert data1.size() == data2.size()
    img_dim = int(data1.size()[-1])
    full_data = data2.view(-1, img_dim).cuda() # (1000, 2048)

    indices = range(len(data1))
    howmany = [1, 5, 20]
    right, total = [0 for x in range(len(howmany))], 0
    for idx in indices:
        vec = data1[idx].view(img_dim).cuda() # [2048]
        ans = cosine_v2m(vec, full_data, True)

        logits_sorted, indexes_2 = torch.sort(ans, dim=0, descending=True)
        indexes_2 = indexes_2.cpu().numpy()
        for ii, how in enumerate(howmany):
            index = indexes_2[:how]
            if idx in index:
                right[ii] += 1
        total += 1
    return right, total

# input : (500 concepts X 15-20 images X 2048 dimension)
# output : (500 concepts X 2048 dimension)
def cnn_mean_helper(data):
    new_data = []
    for data_ in data:
        new_ = torch.mean( data_ , 0 )
        new_data.append(new_)

    return torch.stack(new_data)

# data1, data2 : (500 concepts X 15-20 images X 2048 dimension)
def cnn_mean(data1, data2):
    data1 = cnn_mean_helper(data1)
    data2 = cnn_mean_helper(data2)
    dot = compute_dot_same_shape(data1, data2)
    return dot

# input : (500 concepts X 15-20 images X 2048 dimension)
# output : (500 concepts X 2048 dimension)
def cnn_max_helper(data):
    new_data = []
    for data_ in data:
        new_, _ = torch.max( data_, 0 )
        new_data.append(new_)

    return torch.stack(new_data)

# data1, data2 : (500 concepts X 15-20 images X 2048 dimension)
def cnn_max(data1, data2):
    data1 = cnn_max_helper(data1)
    data2 = cnn_max_helper(data2)
    dot = compute_dot_same_shape(data1, data2)
    return dot

def cosine_v2m(vec, matrix, cosine):
    vec_exp = torch.unsqueeze(vec, 0).expand(matrix.size()) # (bs, 2048)

    prod = torch.mul(vec_exp, matrix) # (bs, 2048)
    prod = torch.sum(prod, 1) # (bs)
    ans = prod

    if cosine:
        norm1 = torch.norm(vec_exp, 2, 1) # (bs)
        norm2 = torch.norm(matrix, 2, 1) # (bs)

        norm = torch.mul(norm1, norm2) # (bs)

        ans = ans / norm # (bs)

    ans = ans.view(-1)
    return ans

def cnn_avgmax(data1, data2, which):
    # data1 : (20, 2048)
    # data2 : (20, 2048)
    ans = [[] for x in range(len(data1))]
    for idx in range(len(data1)): # for each of 20 images
        vec = torch.FloatTensor(data1[idx]).view(-1).cuda()
        ans_ = cosine_v2m(vec, data2, True) # (bs1, 20)

        max_ = torch.max(ans_)
        ans[idx].append(max_)

    ans = torch.FloatTensor(np.array(ans))

    if which == "avgmax":
        return torch.mean(ans)
    elif which == "maxmax":
        return torch.max(ans)

def compute_dot_different_shape(data1, data2, which):
    # 500 X 15-20 X 300
    # list X np.array X np.array

    howmany = [1, 5, 20]
    right, total = [0 for x in range(len(howmany))], 0

    num = len(data1)
    for idx in range( num ):
        sims = []
        vec = data1[idx]

        for idx2 in range(num):
            vec2 = data2[ idx2 ].cuda()
            #vec2 = np.array( data2[ idx2 ] )

            sims.append(cnn_avgmax(vec, vec2, which))

        assert len(sims) == len(data2) # 640
        sims = torch.FloatTensor(np.array(sims)).cuda()

        _, indexes_2 = torch.sort(sims, dim=0, descending=True)
        indexes_2 = indexes_2.cpu().numpy()
        for ii, how in enumerate(howmany):
            index = indexes_2[:how]
            if idx in index:
                right[ii] += 1
        total += 1

        #if idx % 10 == 0:
            #print idx,
    return right, total

def perc(a):
    a0 = np.array(a[0])
    a1 = a[1]
    a0 = a0 / float(a1) * 100
    return "{:.2f} {:.2f} {:.2f}".format(a0[0], a0[1], a0[2])

def perc2(a):
    a0 = np.array(a[0])
    a1 = a[1]
    a0 = a0 / float(a1) * 100
    return np.array( [a0[0], a0[1], a0[2]] )

# l1, l2 : (500 concepts X 15-20 images X 2048 dimension)
def take_pair_helper(l1, l2):
    a1 = cnn_mean(l1, l2)
    print "   cnn_mean : {}".format(a1)
    print "   cnn_mean : {}".format(perc(a1))

    a2 = cnn_max(l1, l2)
    print "   cnn_max : {}".format(a2)
    print "   cnn_max : {}".format(perc(a2))

    a3 = compute_dot_different_shape(l1, l2, "avgmax")
    print "   cnn_avgmax : {}".format(a3)
    print "   cnn_avgmax : {}".format(perc(a3))

    a4 = compute_dot_different_shape(l1, l2, "maxmax")
    print "   max_max : {}".format(a4)
    print "   max_max : {}".format(perc(a4))

    ans = np.array( [ perc2(x) for x in [a1, a2, a3, a4]] )
    return ans.flatten()

def take_pair(l1, l2):
    en_data = torch.load("{}/data/word/{}_2048.pt".format(scr_path(), l1))
    de_data = torch.load("{}/data/word/{}_2048.pt".format(scr_path(), l2))

    keys = en_data.keys()
    keys = np.array([x for x in keys if x in de_data.keys()])
    print "before : {} keys".format(len(keys))
    keys = [k for k in keys if len(en_data[k]) >= 5 and len(de_data[k]) >= 5 ]
    print "after : {} keys".format(len(keys))

    en_data_ = [en_data[k] for k in keys]
    de_data_ = [de_data[k] for k in keys]

    en_train, en_valid = split_data_by_k(en_data_, 5, False)
    de_train, de_valid = split_data_by_k(de_data_, 5, False)
    # list X torch.Tensor
    # 500 X (15-20 X 300)

    print "{} {} keys {} images".format(l1.upper(), len(en_train), sum([len(x) for x in en_train]) )
    print "{} {} keys {} images".format(l2.upper(), len(de_train), sum([len(x) for x in de_train]) )

    return take_pair_helper(en_train, de_train)

if __name__ == '__main__':
    langs = "en de fr es it nl".split()
    results = {}
    for idx in range(len(langs)):
        for idx2 in range(len(langs)):
            if idx < idx2:
                l1, l2 = langs[idx], langs[idx2]
                print l1, l2
                result = take_pair(l1, l2)
                results[l1 + " " + l2] = result
    print results
