# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import argparse
import pickle as pkl
import json
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from pathlib import Path

from util import *
from models import *
from split import split_bergsma
from load_data import *
from forward_funcs import forward_pass_gumbel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation from scratch')

    parser.add_argument("--train_how", type=str, default="gumbel")
    parser.add_argument("--dataset", type=str, default="bergsma")

    parser.add_argument("--l1", type=str, default="en")
    parser.add_argument("--l2", type=str, default="de")

    parser.add_argument("--gpuid", type=int, default=0,
                    help="Which GPU to run")
    parser.add_argument("--num_games", type=int, default=100000000,
                    help="Total number of batches to train for")

    parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size")
    parser.add_argument("--num_dist", type=int, default=2,
                    help="Number of distractors")

    parser.add_argument("--D_img", type=int, default=2048,
                    help="ResNet feature dimensionality")
    parser.add_argument("--D_hid", type=int, default=400,
                    help="Listener's multimodal space dimensionality")

    parser.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.0,
                    help="Speaker hidden layer dropout keep probability")

    parser.add_argument("--temp", type=float, default=1.0,
                    help="Gumbel temperature")
    parser.add_argument("--hard", action="store_true", default=True,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--alpha", type=float, default=1.0,
                    help="Loss = L_spk + alpha * L_lsn")

    parser.add_argument("--no_share_bhd", action="store_true", default=False,
                    help="Do not share beholder. Have a separate visual system for each langauge.")

    parser.add_argument("--print_every", type=int, default=500,
                    help="Print logs every k batches")
    parser.add_argument("--valid_every", type=int, default=500,
                    help="Validate model every k batches")
    parser.add_argument("--translate_every", type=int, default=2500,
                    help="Translate task every k batches")
    parser.add_argument("--save_every", type=int, default=5000,
                    help="Save log data every k batches")

    parser.add_argument("--stop_after", type=int, default=30,
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--lsn_loss_only", action="store_true", default=False,
                    help="re_load pre-trained model")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                    help="Stop if validation loss doesn't improve after k iterations.")

    parser.add_argument("--no_write", action="store_true", default=False,
                    help="re_load pre-trained model")
    parser.add_argument("--no_terminal", action="store_true", default=False,
                    help="re_load pre-trained model")

    parser.add_argument("--cpu", action="store_true", default=False,
                    help="re_load pre-trained model")

    args, remaining_args = parser.parse_known_args()
    if remaining_args != []:
        assert remaining_args == []
    args_dict = vars(args)
    assert args.l1 in "en de es fr it nl".split()
    assert args.l2 in "en de es fr it nl".split()
    assert args.l1 != args.l2

    path = "./{}_{}/".format(args.l1, args.l2)
    model_str = "joint_{}".format(args.train_how)
    if args.lsn_loss_only:
        model_str = "lsn_loss_only." + model_str
    hyperparam_str = "lr_{}.dropout_{}.temp_{}.D_hid_{}.hard_{}.no_share_bhd_{}.num_dist_{}.alpha_{}/".format(args.lr, args.dropout, args.temp, args.D_hid, args.hard, args.no_share_bhd, args.num_dist, args.alpha)
    path_dir = Path(path) / model_str / hyperparam_str

    if not args.no_write and not path_dir.exists():
        path_dir.mkdir(parents=True)

    print args
    print path
    print model_str
    print hyperparam_str

    train_en, valid_en, train_de, valid_de, in_keys, _ = split_bergsma(args.l1, args.l2)
    args.num_cat = len(in_keys)

    in_keys = train_en.keys()
    in_keys = np.array(in_keys)

    print "In keys : {}".format(len(in_keys))

    args.tt = torch if args.cpu else torch.cuda
    model = TwoAgents(args)
    if not args.cpu:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()

    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)
        else:
            print type(param.data), param.size(), "does not require gradients."

    num_params = 0
    for idx in xrange(len(params)):
        num_params += np.prod(params[idx].size())
    print "Number of trainable parameters", num_params

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params, lr=args.lr)

    out_data = {'train':{'x':[], 'y':[] }, \
                'valid':{'x':[], 'y':[] }, \
                'best_valid':{'x':[], 'y':[] }, \
                'translate':{'x':[], 'y':[] } }

    loss_dict = get_log_loss_dict()
    forward_pass = forward_pass_gumbel

    best_val_loss_dict, best_val_epoch, best_val_acc, best_val_loss = None, -1, 0, 999999
    for epoch in xrange(args.num_games):

        final_loss = forward_pass((train_en, train_de), model, loss_dict, args, loss_fn)
        optimizer.zero_grad()
        final_loss.backward()
        total_norm = nn.utils.clip_grad_norm(params, args.grad_clip)
        optimizer.step()

        if epoch >= 0 and epoch % args.print_every == 0:
            avg_loss_dict = get_avg_from_loss_dict(loss_dict)
            print print_loss(epoch, args.alpha, avg_loss_dict, "train")

            out_data['train']['x'].append(epoch)
            out_data['train']['y'].append(avg_loss_dict)
            loss_dict = get_log_loss_dict()

        model.eval()
        if epoch >= 0 and epoch % args.print_every == 0:
            val_loss_dict = get_log_loss_dict()
            for idx in xrange(args.valid_every):
                _ = forward_pass((valid_en, valid_de), model, val_loss_dict, args, loss_fn)

            avg_loss_dict = get_avg_from_loss_dict(val_loss_dict)
            print print_loss(epoch, args.alpha, avg_loss_dict, "valid")
            out_data['valid']['x'].append(epoch)
            out_data['valid']['y'].append(avg_loss_dict)

            cur_loss = np.mean([ avg_loss_dict[agent][role]['loss'] for role in "spk lsn".split() for agent in "agent1 agent2".split() ])
            #cur_acc = np.mean([ avg_loss_dict[agent]['lsn']['acc'] for agent in "agent1 agent2".split() ])
            cur_acc = np.mean([ avg_loss_dict[agent][role]['acc'] for role in "spk lsn".split() for agent in "agent1 agent2".split() ])

            if cur_acc > best_val_acc:
                best_val_epoch = epoch
                best_val_loss = cur_loss
                best_val_acc = cur_acc
                best_val_loss_dict = avg_loss_dict

                out_data['best_valid']['x'] = epoch
                out_data['best_valid']['y'] = avg_loss_dict

                path_model = open( str(path_dir) + "best_model", "wb" )
                torch.save(model.state_dict(), path_model)
                path_model.close()
                print "best model saved."

                result = model.precision(in_keys, args.batch_size)
                out_data['translate']['x'].append(epoch)
                out_data['translate']['y'].append(result)

            if epoch - best_val_epoch >= args.stop_after * args.print_every:
                print "Validation acc not improving after {} iterations.".format(args.stop_after * args.print_every)
                break

        if epoch % args.translate_every == 0:
            model.agent2.listener.nn_words(batch_size=20)
            #result = model.precision(in_keys, args.batch_size)
            #out_data['translate']['x'].append(epoch)
            #out_data['translate']['y'].append(result)

        model.train()
