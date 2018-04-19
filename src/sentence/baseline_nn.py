# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import math
import codecs
import copy
import json
import operator
import cPickle as pkl
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch.utils.serialization import load_lua

from util import *

def cosine_v2v(v1, v2):
    prod = torch.mul(v1, v2) # (bs, 2048)
    prod = torch.sum(prod) # (bs)

    norm1 = torch.norm(v1, 2, 0) # (bs)
    norm2 = torch.norm(v2, 2, 0) # (bs)

    norm = torch.mul(norm1, norm2) # (bs)

    ans = prod / norm # (bs)

    return ans

def cosine_v2m(vec, matrix):
    batch_size = matrix.size()[0]
    vec_exp = vec.view(1, -1).expand(matrix.size()) # (bs, 2048)

    prod = torch.mul(vec_exp, matrix) # (bs, 2048)
    prod = torch.sum(prod, 1) # (bs)
    ans = prod

    norm1 = torch.norm(vec_exp, 2, 1) # (bs)
    norm2 = torch.norm(matrix, 2, 1) # (bs)

    norm = torch.mul(norm1, norm2) # (bs)

    ans = ans / norm # (bs)

    ans = ans.view(batch_size)
    return ans

def rank(data1, data2, batch_size=200, indices=5000):
    # data1 : (5000, 2048)
    # data2 : (55000, 2048)
    len1, len2 = len(data1), len(data2)
    nn_idx = []

    for idx1 in range(indices):
        vec = data1[idx1].view(-1).cuda() # [2048]
        sim_for_vec = []
        for idx2 in range( int( math.ceil( len2 / float(batch_size) ) ) ):
            ini = idx2 * batch_size
            fin = min( (idx2 + 1) * batch_size, len2)
            small_data2 = data2[ini:fin].cuda()

            ans = cosine_v2m(vec, small_data2)

            sim_for_vec.extend(ans.cpu().numpy().tolist())

        assert len(sim_for_vec) == len2

        _, sorted_idx = torch.sort(torch.cuda.FloatTensor(sim_for_vec), dim=0, descending=True)
        sorted_idx = sorted_idx.cpu().numpy().tolist()
        nn_idx.append(sorted_idx[0])

        #if idx1 % 50 == 0:
        #    print "{}/{}".format(idx1, len(data1)),

    return nn_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation from scratch')

    parser.add_argument("--gpuid", type=int, default=0,
                    help="Which GPU to run")
    parser.add_argument("--batch_size", type=int, default=1000,
                    help="Which GPU to run")

    parser.add_argument("--dataset", type=str, default="multi30k",
                    help="Which GPU to run")
    parser.add_argument("--task", type=int, default=2,
                    help="Which GPU to run")
    parser.add_argument("--valid_or_test", type=str, default="test",
                    help="Which GPU to run")

    parser.add_argument("--src", type=str, default="de",
                    help="Which GPU to run")
    parser.add_argument("--trg", type=str, default="en",
                    help="Which GPU to run")

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print args
    assert args.valid_or_test in "valid test".split()

    torch.cuda.set_device(args.gpuid)

    if args.dataset == "coco":
        data_path = coco_path()
        feat_path = coco_path()
        l2 = "jp"
        assert (args.src, args.trg) == ("en", l2) or (args.src, args.trg) == (l2, "en")
    elif args.dataset == "multi30k":
        feat_path = multi30k_reorg_path()
        data_path = multi30k_reorg_path() + "task{}/".format(args.task)
        l2 = "de"
        assert (args.src, args.trg) == ("en", l2) or (args.src, args.trg) == (l2, "en")

    (train_en_img, train_l2_img, valid_img, test_img) = [torch.load('{}/half_feats/{}'.format(feat_path, x)) \
        for x in "train_en_feats train_{}_feats valid_feats test_feats".format(l2).split() ]
    (en_w2i, en_i2w, l2_w2i, l2_i2w) = [torch.load('{}/dics/{}'.format(data_path, x)) \
        for x in "en_w2i en_i2w {}_w2i {}_i2w".format(l2, l2).split()]
    (train_en_org, test_en_org, train_l2_org, test_l2_org) = [torch.load('{}/half_labs/{}'.format(data_path, x)) \
        for x in "en_train_org en_test_org {}_train_org {}_test_org".format(l2, l2).split()]

    print "{} : {}-{} translation".format(args.dataset, args.src, args.trg)
    print "{} {} caption \n\t\t\t-> {} {} image \n\t\t\t-> closest {} training image \n\t\t\t-> corresponding {} training caption \n\t\t\t-> compute BLEU against {} {} caption".format(\
               args.src.upper(), args.valid_or_test, args.src.upper(), args.valid_or_test, args.trg.upper(), args.trg.upper(), args.trg.upper(), args.valid_or_test)

    if args.valid_or_test == "valid":
        cmp_from = valid_img
    elif args.valid_or_test == "test":
        cmp_from = test_img

    if args.src == "en":
        cmp_to = train_l2_img
        #src_text = test_en_org
        trg_text = train_l2_org
        i2w = l2_i2w

    elif args.trg == "en":
        cmp_to = train_en_img
        #src_text = test_l2_org
        trg_text = train_en_org
        i2w = en_i2w

    #assert len(cmp_from) == len(src_text)
    print "comparing {} {} {} images with {} {} training images".format(len(cmp_from), args.src.upper(), args.valid_or_test, len(cmp_to), args.trg.upper())
    num = 1 if (args.dataset == "multi30k" and args.task == 1) else 5

    indices = len(cmp_from)
    nn_idx = rank(cmp_from, cmp_to, batch_size=args.batch_size, indices=indices)
    assert len(nn_idx) == len(cmp_from)

    trg_hyp = [trg_text[idx][:num] for idx in nn_idx]
    trg_hyp = [ [" ".join( [ i2w[each_word] for each_word in each_hyp[1:-1] ] ).replace("@@ ", "") for each_hyp in five_hyps ] for five_hyps in trg_hyp]
    final_ans = []
    for five_hyps in trg_hyp:
        for each in five_hyps:
            final_ans.append(each.strip().replace('\n',''))

    dest_path = '{}/nn_baseline/{}2{}_{}'.format(data_path, args.src, args.trg, args.valid_or_test)
    dest = codecs.open(dest_path, 'wb', encoding="utf8")
    dest.write( u'\r\n'.join( final_ans ) )
    dest.close()

    print "{} {}-{} {}".format(args.dataset, args.src, args.trg, args.valid_or_test)
    command = 'perl {}/multi-bleu.perl {} < {}'.format(scr_path(), '{}/ref/{}_many_{}'.format(data_path, args.trg, args.valid_or_test), dest_path )
    os.system(command)

