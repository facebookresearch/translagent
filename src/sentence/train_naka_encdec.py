# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import ipdb
import sys
import codecs
import copy
import commands
import argparse
import math
import pickle as pkl
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from util import *
from models import NakaAgent
from forward import forward_naka
from dataloader import *

random = np.random
random.seed(1234)

def translate(args, agent, labels, i2w, batch_size, which, tt):
    src_lab_org, trg_lab_org = labels[args.src], labels[args.trg]
    image_ids = random.choice( range(len(src_lab_org)), batch_size, replace=False ) # (num_dist)
    src_cap_ids = [random.randint(0, len(src_lab_org[ image_ids[idx] ])) for idx in range(batch_size)  ]  # choose an object
    trg_cap_ids = [random.randint(0, len(trg_lab_org[ image_ids[idx] ])) for idx in range(batch_size)  ]  # choose an object

    src_caps = np.array([src_lab_org[image_id][caption_id] for (image_id, caption_id) in zip(image_ids, src_cap_ids)])
    trg_caps = np.array([trg_lab_org[image_id][caption_id] for (image_id, caption_id) in zip(image_ids, trg_cap_ids)])

    src_sorted_idx = sort_per_len(src_caps)

    src_caps = [ x[1:-1] for x in src_caps[src_sorted_idx] ]
    trg_caps = [ x[1:-1] for x in trg_caps[src_sorted_idx] ]

    l2_src = print_captions(src_caps, i2w[args.src], " ")
    en_ref = print_captions(trg_caps, i2w[args.trg], " ")
    en_hyp = agent.translate(src_caps)

    print "---------------- {} TRANSLATION ----------------".format(which)
    for idx in range(len(en_hyp)):
        print u"{} src {} | {}".format(args.src.upper(), idx+1, l2_src[idx] )
        print u"{} ref {} | {}".format(args.trg.upper(), idx+1, en_ref[idx].strip() )
        print u"{} hyp {} | {}".format(args.trg.upper(), idx+1, en_hyp[idx] )
        print ""
    print "---------------------------------------------"

def valid_bleu(labels, model, args, tt, dir_dic, decode_how, which_dataset="valid"):
    num = 1 if (args.dataset == "multi30k" and args.task == 1) else 5
    src = labels[args.src]
    batch_size = 200
    num_imgs = len(src)
    model_gen = [[] for x in range(num)]
    for cap_idx in range(num):
        for batch_idx in range( int( math.ceil( float(num_imgs) / batch_size ) ) ):
            start_idx = batch_idx * batch_size
            end_idx = min( num_imgs, (batch_idx + 1) * batch_size )
            l2_caps = np.array([src[img_idx][cap_idx][1:-1] for img_idx in range(start_idx, end_idx)])

            l2_cap_lens = sort_per_len(l2_caps)
            inverse = np.argsort(l2_cap_lens)

            l2_caps = l2_caps[l2_cap_lens]
            en_hyp = model.translate(l2_caps, decode_how)
            en_hyp = [en_hyp[idx] for idx in inverse]
            model_gen[cap_idx].extend( en_hyp )

    final_out = []
    for idx in range(num_imgs):
        for i2 in range(num):
            final_out.append(model_gen[i2][idx])

    destination = dir_dic["path_dir"] + "{}_hyp_{}".format(which_dataset, decode_how)
    f = codecs.open(destination, 'wb', encoding="utf8")
    f.write( u'\r\n'.join( final_out ) )
    f.close()

    command = 'perl {}/multi-bleu.perl {} < {}'.format(scr_path(), '{}/ref/{}_many_{}'.format(dir_dic["data_path"], args.trg, which_dataset), destination  )

    bleu = commands.getstatusoutput(command)[1]
    print which_dataset, bleu[ bleu.find("BLEU"): ]
    bleu_score = float(bleu[ bleu.find("=")+1: bleu.find(",", bleu.find("=")+1) ] )
    return bleu_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation from scratch')

    parser.add_argument("--gpuid", type=int, default=0,
                    help="Which GPU to run")

    parser.add_argument("--lmbd", type=float, default=10,
                    help="Which GPU to run")
    parser.add_argument("--margin", type=float, default=0.1,
                    help="Which GPU to run")

    parser.add_argument("--two_fc", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--dataset", type=str, default="multi30k",
                    help="Which GPU to run")
    parser.add_argument("--task", type=int, default=1,
                    help="Which GPU to run")
    parser.add_argument("--loss_type", type=str, default="margin",
                    help="Which GPU to run")

    parser.add_argument("--beam_width", type=int, default=2,
                    help="Which GPU to run")
    parser.add_argument("--norm_pow", type=float, default=1.0,
                    help="Which GPU to run")

    parser.add_argument("--train_enc_how", type=str,  # NOTE beam / greedy
                    help="Which GPU to run")
    parser.add_argument("--train_dec_how", type=str,  # NOTE beam / greedy
                    help="Which GPU to run")

    parser.add_argument("--src", type=str,  # NOTE beam / greedy
                    help="Which GPU to run")
    parser.add_argument("--trg", type=str,  # NOTE beam / greedy
                    help="Which GPU to run")

    parser.add_argument("--num_games", type=int, default=10000000000,
                    help="Total number of batches to train for")
    parser.add_argument("--batch_size", type=int, default=100,
                    help="Batch size")

    parser.add_argument("--D_img", type=int, default=2048,
                    help="ResNet feature dimensionality")
    parser.add_argument("--D_hid", type=int, default=1024,
                    help="Token embedding dimensionality")
    parser.add_argument("--D_emb", type=int, default=512,
                    help="Token embedding dimensionality")
    parser.add_argument("--num_layers", type=int, default=1,
                    help="Token embedding dimensionality")

    parser.add_argument("--seq_len_en", type=int, default=80,
                    help="Token embedding dimensionality")
    parser.add_argument("--seq_len_l2", type=int, default=80,
                    help="Token embedding dimensionality")

    parser.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout keep probability")

    parser.add_argument("--drop_img", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--drop_bhd", action="store_true", default=True,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--drop_emb", action="store_true", default=True,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--drop_out", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--temp", type=float, default=0.3,
                    help="Gumbel temperature")
    parser.add_argument("--hard", action="store_true", default=True,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--print_every", type=int, default=100,
                    help="Save model output.")
    parser.add_argument("--valid_every", type=int, default=500,
                    help="Validate model every k batches")
    parser.add_argument("--sample_every", type=int, default=2000,
                    help="Validate model every k batches")
    parser.add_argument("--save_every", type=int, default=2000,
                    help="Save model output.")
    parser.add_argument("--translate_every", type=int, default=2000,
                    help="Save model output.")

    parser.add_argument("--stop_after", type=int, default=30,
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--no_share_bhd", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--cpu", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--unit_norm", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--seq_len", type=int, default=80,
                    help="Stop if validation loss doesn't improve after k iterations.")

    parser.add_argument("--no_write", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--no_terminal", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)

    assert (args.train_enc_how == "two" and args.train_dec_how == "img") or (args.train_enc_how == "three" and args.train_dec_how in "img des both".split() )
    assert args.loss_type in "margin reciprocal".split()
    assert args.trg != None and args.src != None

    args.to_drop = []
    if args.drop_img:
        args.to_drop.append("img")
    if args.drop_emb:
        args.to_drop.append("emb")
    if args.drop_out:
        args.to_drop.append("out")
    if args.drop_bhd:
        args.to_drop.append("bhd")
    args.to_drop.sort()

    if args.loss_type == "margin":
        args.unit_norm = True
        print "Unit norm"

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

    (train_img_en, train_img_l2, valid_img, test_img) = [torch.load('{}/half_feats/{}'.format(feat_path, x)) \
        for x in "train_en_feats train_{}_feats valid_feats test_feats".format(args.l2).split() ]

    (w2i_en, i2w_en, w2i_l2, i2w_l2) = [torch.load(data_path + 'dics/{}'.format(x)) \
        for x in "{}_w2i {}_i2w {}_w2i {}_i2w".format("en", "en", args.l2, args.l2).split()]

    (train_org_en, valid_org_en, test_org_en) = [torch.load(data_path + 'half_labs/{}'.format(x)) \
        for x in "{}_train_org {}_valid_org {}_test_org".format("en", "en", "en").split()]
    (train_org_l2, valid_org_l2, test_org_l2) = [torch.load(data_path + 'half_labs/{}'.format(x)) \
        for x in "{}_train_org {}_valid_org {}_test_org".format(args.l2, args.l2, args.l2).split()]

    train_org_en = trim_caps(train_org_en, 4, args.seq_len_en)
    train_org_l2 = trim_caps(train_org_l2, 4, args.seq_len_l2)
    args.vocab_size_en = len(w2i_en)
    args.vocab_size_l2 = len(w2i_l2)

    path = "{}sentence_level/{}/naka_joint/{}-{}/".format(saved_results_path(), task_path, args.src, args.trg, )
    model_str = "{}_loss.{}-way.train_dec_{}/".format(args.loss_type, args.train_enc_how, args.train_dec_how)

    hyperparam_str = "lambda_{}.lr_{}.dropout_{}.D_hid_{}.D_emb_{}.vocab_size_{}_{}.unit_norm_{}/".format(args.lmbd, args.lr, args.dropout, args.D_hid, args.D_emb, args.vocab_size_en, args.vocab_size_l2, args.unit_norm)
    path_dir = path + model_str + hyperparam_str
    if not args.no_write:
        recur_mkdir(path_dir)

    sys.stdout = Logger(path_dir, no_write=args.no_write, no_terminal=args.no_terminal)
    print args
    print path
    print model_str
    print hyperparam_str
    dir_dic = {"feat_path":feat_path, "data_path":data_path, "task_path":task_path, "path":path, "path_dir":path_dir}

    train_images = { "en":train_img_en, args.l2:train_img_l2 }
    valid_images = { "en":valid_img, args.l2:valid_img }
    train_labels = { "en":train_org_en, args.l2:train_org_l2 }
    valid_labels = { "en":valid_org_en, args.l2:valid_org_l2 }
    test_labels = { "en":test_org_en, args.l2:test_org_l2 }

    args.vocab_size = {"en":args.vocab_size_en, args.l2:args.vocab_size_l2}
    args.num_layers = {"spk" : { "en":1, args.l2:1 }, \
                  "lsn" : { "en":1, args.l2:1} }
    args.num_directions = {"lsn" : { "en":1, args.l2:1 } }
    args.w2i = { "en":w2i_en, args.l2:w2i_l2 }
    args.i2w = { "en":i2w_en, args.l2:i2w_l2 }
    args.seq_len = { "en":args.seq_len_en, args.l2:args.seq_len_l2 }

    loss_fn = nn.CrossEntropyLoss()
    model = NakaAgent(args.trg, args.src, args)
    if not args.cpu:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()
        loss_fn = loss_fn.cuda()
        tt = torch.cuda

    in_params, out_params = [], []
    in_names, out_names = [], []

    for name, param in model.named_parameters():
        in_params.append(param)
        in_names.append(name)

    in_size, out_size = [x.size() for x in in_params], [x.size() for x in out_params]
    in_sum, out_sum = sum([np.prod(x) for x in in_size]), sum([np.prod(x) for x in out_size])

    print "IN    : {} params".format(in_sum)
    print print_params_naka(in_names, in_size)
    print "OUT   : {} params".format(out_sum)
    print print_params_naka(out_names, out_size)
    print "TOTAL : {} params".format(in_sum + out_sum)

    out_data = {'train':{'x':[], 'y':[] }, \
                'valid':{'x':[], 'y':[] }, \
                'bleu':{'x':[], 'y':[] }, \
                'best_valid':{'x':[], 'y':[] } }

    optimizer = torch.optim.Adam(in_params, lr=args.lr)
    train_losses = {"enc_src": AverageMeter(), "enc_trg": AverageMeter(), "dec_img": AverageMeter(), "dec_des": AverageMeter()}
    best_epoch = -1
    best_bleu = {"valid":-1, "test":-1}

    for epoch in xrange(args.num_games):
        curr_loss = forward_naka( train_images, train_labels, model, train_losses, args, loss_fn, tt )
        optimizer.zero_grad()
        curr_loss.backward()
        total_norm = nn.utils.clip_grad_norm(in_params, args.grad_clip)
        optimizer.step()

        if epoch % args.print_every == 0:
            print "epoch {} train | enc_src {:.3f} enc_trg {:.3f} dec_img {:.3f} dec_des {:.3f} ".format(epoch, train_losses["enc_src"].avg, train_losses["enc_trg"].avg, \
                                                                                                         train_losses["dec_img"].avg, train_losses["dec_des"].avg,)
            out_data['train']['x'].append(epoch)
            out_data['train']['y'].append(train_losses)
            train_losses['enc_src'].reset(); train_losses['enc_trg'].reset(); train_losses['dec_img'].reset(); train_losses['dec_des'].reset();

        model.eval()
        if epoch > 0 and epoch % args.valid_every == 0:
            valid_losses = {"enc_src": AverageMeter(), "enc_trg": AverageMeter(), "dec_img": AverageMeter(), "dec_des": AverageMeter()}
            for idx in range(args.print_every):
                _ = forward_naka( valid_images, valid_labels, model, valid_losses, args, loss_fn, tt )

            out_data['valid']['x'].append(epoch)
            out_data['valid']['y'].append(valid_losses)

            bleu_score = valid_bleu(valid_labels, model, args, tt, dir_dic, "greedy", "valid")

            print "epoch {} valid | enc_src {:.3f} enc_trg {:.3f} dec_img {:.3f} dec_des {:.3f} | bleu {}".format(epoch, valid_losses["enc_src"].avg, valid_losses["enc_trg"].avg, \
                                                                                                         valid_losses["dec_img"].avg, valid_losses["dec_des"].avg, bleu_score)

            if bleu_score > best_bleu["valid"]:
                best_bleu["valid"] = bleu_score
                best_epoch = epoch
                path_model = open( path_dir + "best_model.pt", "wb" )
                torch.save(model.state_dict(), path_model)
                path_model.close()
                print "best model saved : {} BLEU".format(bleu_score)
                test_bleu = valid_bleu(test_labels, model, args, tt, dir_dic, "greedy", "test")
                best_bleu["test"] = test_bleu
            else:
                if epoch - best_epoch >= args.stop_after * args.valid_every:
                    print "Validation BLEU not improving after {} iterations, early stopping".format( args.stop_after * args.valid_every )
                    print "Best BLEU {}".format(best_bleu)
                    break

        if epoch % args.translate_every == 0:
            translate(args, model, valid_labels, args.i2w, 5, "VALID", tt)

        if epoch > 0 and epoch % args.save_every == 0:
            path_results = open( path_dir + "results", "w" )
            pkl.dump(out_data, path_results)
            path_results.close()
            print "results saved."

        model.train()

    path_results = open( path_dir + "results", "w" )
    pkl.dump(out_data, path_results)
    path_results.close()
    print "results saved."

    result = open( path + model_str + "result", "a")
    result.write( "{}\t{}\t{}\n".format(best_bleu['valid'], best_bleu['test'], hyperparam_str) )
    result.close()
