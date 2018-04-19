# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import commands
import codecs
import copy
import argparse
import math
import pickle as pkl
import os
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.serialization import load_lua

from util import *
from models import *
from dataloader import *
from forward import *

random = np.random
random.seed(1234)

def translate(model, src, trg, image, valid_labs, i2w, args, tt ):
    num = 1 if (args.dataset == "multi30k" and args.task == 1) else 5
    if trg == "en":
        agent = model.en_agent
    else:
        agent = model.l2_agent

    lab_org = valid_labs[src]
    i2w_src, i2w_trg = i2w[src], i2w[trg]

    keys = range(len(lab_org))
    img_indices = np.array(random.choice( keys, 4, replace=False ))
    src_caps = np.array([lab_org[ind][ np.random.randint( num ) ][1:-1] for ind in img_indices])

    src_cap_lens = sort_per_len(src_caps)
    src_caps, img_indices = src_caps[src_cap_lens], img_indices[src_cap_lens]

    img = Variable(image.index_select(0, torch.LongTensor(img_indices)), requires_grad=False)
    if tt == torch.cuda:
        img = img.cuda()
    h_img = agent.beholder1(img) if args.no_share_bhd else agent.beholder(img)
    spk_output = agent.speaker.sample(h_img, True)
    spk_output = decode(spk_output, i2w_trg)

    l2_src = decode(src_caps, i2w_src)
    en_hyp = agent.translate(src_caps, "greedy")
    print "---------------- {}-{} ----------------".format(src.upper(), trg.upper())
    for idx in range(len(en_hyp)):
        print u"{} src {} | {}".format(src.upper(), idx+1, l2_src[idx] )
        print u"{} cap {} | {}".format(trg.upper(), idx+1, spk_output[idx] )
        print u"{} hyp {} | {}".format(trg.upper(), idx+1, en_hyp[idx] )
        print ""
    print "---------------------------------------------"

def valid_bleu(valid_labels, model, args, tt, dir_dic, which_dataset="valid"):
    batch_size = 200
    num = 1 if (args.dataset == "multi30k" and args.task == 1) else 5
    bleu_dic = {}
    for langs in "en_{} {}_en".format(args.l2, args.l2).split():
        (src, trg) = langs.split("_")
        if trg == "en":
            agent = model.en_agent
        elif src == "en":
            agent = model.l2_agent

        num_imgs = len(valid_labels[src])
        model_gen = [[] for x in range(num)]

        for cap_idx in range(num):
            for batch_idx in range( int( math.ceil( float(num_imgs) / args.batch_size ) ) ):
                start_idx = batch_idx * args.batch_size
                end_idx = min( num_imgs, (batch_idx + 1) * args.batch_size )

                src_caps = np.array([valid_labels[src][img_idx][cap_idx][1:-1] for img_idx in range(start_idx, end_idx)])
                src_cap_lens = sort_per_len(src_caps)
                inverse = np.argsort(src_cap_lens)

                src_caps = src_caps[src_cap_lens]
                trg_hyp = agent.translate(src_caps, decode_how=args.decode_how)
                trg_hyp = [trg_hyp[idx] for idx in inverse]
                model_gen[cap_idx].extend( trg_hyp )

        final_out = []
        for idx in range(num_imgs):
            for i2 in range(num):
                final_out.append(model_gen[i2][idx])

        destination = dir_dic["path_dir"] + "{}-{}_{}_hyp_{}".format(src, trg, which_dataset, args.decode_how)
        f = codecs.open(destination, 'wb', encoding="utf8")
        f.write( u'\r\n'.join( final_out ) )
        f.close()

        command = 'perl {}/multi-bleu.perl {} < {}'.format(scr_path(), '{}/ref/{}_many_{}'.format(dir_dic["data_path"], trg, which_dataset), destination )

        bleu = commands.getstatusoutput(command)[1]
        print which_dataset, langs, bleu[ bleu.find("BLEU"): ]
        bleu_score = float(bleu[ bleu.find("=")+1: bleu.find(",", bleu.find("=")+1) ] )
        bleu_dic[ langs ] = bleu_score

    return bleu_dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation from scratch')

    parser.add_argument("--gpuid", type=int, default=0,
                    help="Which GPU to run")

    parser.add_argument("--dataset", type=str, default="multi30k",
                    help="Which GPU to run")
    parser.add_argument("--task", type=int, default=2,
                    help="Which GPU to run")

    parser.add_argument("--alpha", type=float, default=1.0,
                    help="Which GPU to run")

    parser.add_argument("--two_fc", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--num_games", type=int, default=10000000000,
                    help="Total number of batches to train for")
    parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size")
    parser.add_argument("--num_dist", type=int, default=2,
                    help="Batch size")

    parser.add_argument("--D_img", type=int, default=2048,
                    help="ResNet feature dimensionality")
    parser.add_argument("--D_hid", type=int, default=512,
                    help="Token embedding dimensionality")
    parser.add_argument("--D_emb", type=int, default=256,
                    help="Token embedding dimensionality")

    parser.add_argument("--seq_len_en", type=int, default=80,
                    help="Token embedding dimensionality")
    parser.add_argument("--seq_len_l2", type=int, default=80,
                    help="Token embedding dimensionality")

    parser.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout keep probability")

    parser.add_argument("--temp", type=float, default=1.0,
                    help="Gumbel temperature")
    parser.add_argument("--hard", action="store_true", default=True,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--print_every", type=int, default=100,
                    help="Save model output.")
    parser.add_argument("--valid_every", type=int, default=500,
                    help="Validate model every k batches")
    parser.add_argument("--translate_every", type=int, default=2000,
                    help="Validate model every k batches")
    parser.add_argument("--save_every", type=int, default=4000,
                    help="Save model output.")

    parser.add_argument("--stop_after", type=int, default=30,
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                    help="Stop if validation loss doesn't improve after k iterations.")

    parser.add_argument("--unit_norm", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--cpu", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--pretrain_spk", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--which_loss", type=str, default="joint",
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--loss_type", type=str, default="xent",
                    help="Stop if validation loss doesn't improve after k iterations.")

    parser.add_argument("--fix_spk", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--fix_bhd", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--no_share_bhd", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--decode_how", type=str, default="greedy",
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--sample_how", type=str, default="gumbel",
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--beam_width", type=int, default=2,
                    help="Which GPU to run")
    parser.add_argument("--norm_pow", type=float, default=1.0,
                    help="Which GPU to run")

    parser.add_argument("--re_load", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--no_write", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--no_terminal", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)

    if args.dataset == "coco":
        feat_path = coco_path()
        data_path = coco_path()
        task_path = args.dataset
        args.l2 = "jp"

    elif args.dataset == "multi30k":
        feat_path = multi30k_reorg_path()
        data_path = multi30k_reorg_path() + "task{}/".format(args.task)
        task_path = "{}_task{}".format(args.dataset, args.task)
        args.l2 = "de"

    (train_img1, train_img2, valid_img, test_img) = [torch.load('{}/half_feats/{}'.format(feat_path, x)) \
        for x in "train_en_feats train_{}_feats valid_feats test_feats".format(args.l2).split() ]

    (w2i_en, i2w_en, w2i_l2, i2w_l2) = [torch.load(data_path + 'dics/{}'.format(x)) \
        for x in "{}_w2i {}_i2w {}_w2i {}_i2w".format("en", "en", args.l2, args.l2).split()]

    (train_en1, valid_en, test_en) = [torch.load(data_path + 'half_labs/{}'.format(x)) \
        for x in "{}_train_org {}_valid_org {}_test_org".format("en", "en", "en").split()]
    (train_l22, valid_l2, test_l2) = [torch.load(data_path + 'half_labs/{}'.format(x)) \
        for x in "{}_train_org {}_valid_org {}_test_org".format(args.l2, args.l2, args.l2).split()]

    if args.dataset == "coco":
        a, b = get_coco_idx()
        train_img1 = train_img1.index_select(0, torch.LongTensor(a))
        train_img2 = train_img2.index_select(0, torch.LongTensor(b))

        train_en1 = [train_en1[idx] for idx in a]
        train_l21 = [train_l21[idx] for idx in a]

        train_en2 = [train_en2[idx] for idx in b]
        train_l22 = [train_l22[idx] for idx in b]

        print train_img1.size(), train_img2.size()
        print len(train_en1), len(train_l21), len(train_en2), len(train_l22)

    args.vocab_size_en = len(w2i_en)
    args.vocab_size_l2 = len(w2i_l2)

    train_en = trim_caps(train_en1, 4, args.seq_len_en)
    train_l2 = trim_caps(train_l22, 4, args.seq_len_l2)

    fixed, learned = [], ["lsn"]

    if args.fix_spk:
        fixed.append("spk")
    else:
        learned.append("spk")

    if args.fix_bhd:
        fixed.append("bhd")
    else:
        learned.append("bhd")

    fixed, learned = "_".join(sorted(fixed)), "_".join(sorted(learned))

    assert args.which_loss in "joint lsn".split()
    model_str = "fixed_{}.learned_{}.{}_loss/".format(fixed, learned, args.which_loss)
    if args.pretrain_spk:
        model_str = "pretrain_spk." + model_str
    if args.no_share_bhd:
        model_str = "no_share_bhd." + model_str

    mill = int(round(time.time() * 1000)) % 1000

    big = "{}sentence_level/{}".format(saved_results_path(), task_path)
    path = "{}sentence_level/{}/joint_model/".format(saved_results_path(), task_path)
    hyperparam_str = "{}_dropout_{}.alpha_{}.lr_{}.temp_{}.D_hid_{}.D_emb_{}.num_dist_{}.vocab_size_{}_{}.hard_{}/".format(mill, args.dropout, args.alpha, args.lr, args.temp, args.D_hid, args.D_emb, args.num_dist, args.vocab_size_en, args.vocab_size_l2, args.hard )
    path_dir = path + model_str + hyperparam_str
    if not args.no_write:
        recur_mkdir(path_dir)

    sys.stdout = Logger(path_dir, no_write=args.no_write, no_terminal=args.no_terminal)
    print args
    print model_str
    print hyperparam_str
    dir_dic = {"feat_path":feat_path, "data_path":data_path, "task_path":task_path, "path":path, "path_dir":path_dir}

    args.vocab_size = {"en":args.vocab_size_en, args.l2:args.vocab_size_l2}
    args.num_layers = {"spk" : { "en":1, args.l2:1 }, \
                  "lsn" : { "en":1, args.l2:1} }
    args.num_directions = {"lsn" : { "en":1, args.l2:1 } }
    args.w2i = { "en":w2i_en, args.l2:w2i_l2 }
    args.i2w = { "en":i2w_en, args.l2:i2w_l2 }
    args.seq_len = { "en":args.seq_len_en, args.l2:args.seq_len_l2 }

    train_images = { "en":train_img1, args.l2:train_img2 }
    valid_images = { "en":valid_img, args.l2:valid_img }

    train_labels = { "en":train_en, args.l2:train_l2 }
    valid_labels = { "en":valid_en, args.l2:valid_l2 }
    test_labels = { "en":test_en, args.l2:test_l2 }

    model = TwoAgents(args)
    print model
    if not args.cpu:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()

    in_params, out_params = [], []
    in_names, out_names = [], []
    for name, param in model.named_parameters():
        if ("speaker" in name and args.fix_spk) or\
           ("beholder" in name and args.fix_bhd):
            out_params.append(param)
            out_names.append(name)
        else:
            in_params.append(param)
            in_names.append(name)

    in_size, out_size = [x.size() for x in in_params], [x.size() for x in out_params]
    in_sum, out_sum = sum([np.prod(x) for x in in_size]), sum([np.prod(x) for x in out_size])

    print "IN    : {} params".format(in_sum)
    #print print_params(in_names, in_size)
    print "OUT   : {} params".format(out_sum)
    #print print_params(out_names, out_size)
    print "TOTAL : {} params".format(in_sum + out_sum)

    loss_fn = {'xent':nn.CrossEntropyLoss(), 'mse':nn.MSELoss(), 'mrl':nn.MarginRankingLoss(), 'mlml':nn.MultiLabelMarginLoss(), 'mml':nn.MultiMarginLoss()}
    tt = torch
    if not args.cpu:
        loss_fn = {k:v.cuda() for (k,v) in loss_fn.iteritems()}
        tt = torch.cuda

    optimizer = torch.optim.Adam(in_params, lr=args.lr)

    out_data = {'train':{'x':[], 'y':[] }, \
                'valid':{'x':[], 'y':[] }, \
                'bleu':{'x':[], 'y':[] }, \
                'best_valid':{'x':[], 'y':[] } }

    best_epoch = -1
    best_bleu = {"valid":{0:0}, "test":{0:0}}

    train_loss_dict = get_log_loss_dict()
    for epoch in xrange(args.num_games):
        loss = forward_joint(train_images, train_labels, model, train_loss_dict, args, loss_fn, args.num_dist, tt)
        optimizer.zero_grad()
        loss.backward()
        total_norm = nn.utils.clip_grad_norm(in_params, args.grad_clip)
        optimizer.step()

        if epoch % args.print_every == 0:
            avg_loss_dict = get_avg_from_loss_dict(train_loss_dict)
            print print_loss(epoch, args.alpha, avg_loss_dict, "train")

            out_data['train']['x'].append(epoch)
            out_data['train']['y'].append(avg_loss_dict)
            train_loss_dict = get_log_loss_dict()

        model.eval()

        if epoch % args.valid_every == 0:
            valid_loss_dict = get_log_loss_dict()
            for idx in range(args.print_every):
                _ = forward_joint(valid_images, valid_labels, model, valid_loss_dict, args, loss_fn, args.num_dist, tt)

            avg_loss_dict = get_avg_from_loss_dict(valid_loss_dict)
            print print_loss(epoch, args.alpha, avg_loss_dict, "valid")
            out_data['valid']['x'].append(epoch)
            out_data['valid']['y'].append(avg_loss_dict)

            if not args.no_write:
                bleu_dic = valid_bleu(valid_labels, model, args, tt, dir_dic, "valid")
                out_data['bleu']['x'].append(epoch)
                out_data['bleu']['y'].append(bleu_dic)

                if np.mean( bleu_dic.values() ) > np.mean( best_bleu["valid"].values() ):
                    best_bleu["valid"] = bleu_dic
                    best_epoch = epoch
                    path_model = open( path_dir + "best_model.pt", "wb" )
                    torch.save(model.state_dict(), path_model)
                    path_model.close()
                    print "best model saved : {} BLEU".format(bleu_dic)
                    test_bleu_dic = valid_bleu(test_labels, model, args, tt, dir_dic, "test")
                    best_bleu["test"] = test_bleu_dic
                else:
                    if epoch - best_epoch >= args.stop_after * args.valid_every:
                        print "Validation BLEU not improving after {} iterations, early stopping".format( args.stop_after * args.valid_every )
                        print "Best BLEU {}".format(best_bleu)
                        break

        if epoch % args.translate_every == 0:
            translate(model, "en", args.l2, valid_img, valid_labels, args.i2w, args, tt)
            translate(model, args.l2, "en", valid_img, valid_labels, args.i2w, args, tt)

        if epoch > 0 and epoch % args.save_every == 0:
            if not args.no_write:
                path_results = open( path_dir + "results", "wb" )
                pkl.dump(out_data, path_results)
                path_results.close()
            print "results saved."

        model.train()
