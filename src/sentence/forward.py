# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
import random
import pickle as pkl
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

from util import idx_to_emb, logit_to_acc

from dataloader import next_batch_joint, next_batch_naka_enc, next_batch_nmt

millis = int(round(time.time() * 10000)) % 10000
random.seed(millis)

def forward_nmt(labels, model, loss_dict, args, loss_fn, tt):
    src_caps_in, src_caps_in_lens, trg_sorted_idx, trg_caps_in, trg_caps_in_lens, trg_caps_out = next_batch_nmt(labels[args.src], labels[args.trg], args.batch_size, tt)
    dec_logits = model(src_caps_in, src_caps_in_lens, trg_sorted_idx, trg_caps_in, trg_caps_in_lens)

    loss = loss_fn['xent'](dec_logits, trg_caps_out)
    loss_dict['loss'].update(loss.data[0])
    return loss

def forward_joint(images, labels, model, loss_dict, args, loss_fn, num_dist, tt):
    en_batch = next_batch_joint(images['en'], labels['en'], args.batch_size, num_dist, tt)
    l2_batch = next_batch_joint(images[args.l2], labels[args.l2], args.batch_size, num_dist, tt)
    #spk_imgs, lsn_imgs, spk_caps_in, spk_cap_lens, _, _, spk_caps_out, whichs

    # spk_imgs : (batch_size, 2048)
    # lsn_imgs : (batch_size, num_dist, 2048)

    output_en, output_l2, comm_actions = model(en_batch[:4], l2_batch[:4], args.sample_how)
    # output_en : (en_spk_logits, (en_rnn_hid, en_lsn_h_imgs) )
    # output_l2 : (l2_spk_logits, (l2_rnn_hid, l2_lsn_h_imgs) )
    # comm_actions : (batch_size, seq_len), already masked
    final_loss = 0

    en_spk_loss = loss_fn['xent'](output_en[0], en_batch[6])
    l2_spk_loss = loss_fn['xent'](output_l2[0], l2_batch[6])
    loss_dict["l1"]["spk"]["loss"].update(en_spk_loss.data[0])
    loss_dict["l2"]["spk"]["loss"].update(l2_spk_loss.data[0])

    if args.which_loss == "joint":
        final_loss += en_spk_loss * args.alpha
        final_loss += l2_spk_loss * args.alpha

    if args.loss_type == "xent":
        en_diff_dist = torch.mean( torch.pow(output_en[1][0] - output_en[1][1], 2), 2).view(-1, args.num_dist)
        en_logits = 1 / (en_diff_dist + 1e-10)
        en_lsn_loss = loss_fn['xent']( en_logits, en_batch[7] )
        en_lsn_acc = logit_to_acc(en_logits, en_batch[7]) * 100
        final_loss += en_lsn_loss

        l2_diff_dist = torch.mean( torch.pow(output_l2[1][0] - output_l2[1][1], 2), 2).view(-1, args.num_dist)
        l2_logits = 1 / (l2_diff_dist + 1e-10)
        l2_lsn_loss = loss_fn['xent']( l2_logits, l2_batch[7] )
        l2_lsn_acc = logit_to_acc(l2_logits, l2_batch[7]) * 100
        final_loss += l2_lsn_loss

    elif args.loss_type == "mse":
        en_diff_dist = torch.mean( torch.pow(output_en[1][0] - output_en[1][1], 2), 2).view(-1, args.num_dist)
        en_logits = 1 / (en_diff_dist + 1e-10)
        en_lsn_acc = logit_to_acc(en_logits, en_batch[7]) * 100

        en_diff_dist = torch.masked_select(en_diff_dist, idx_to_emb( en_batch[7].cpu().data.numpy(), args.num_dist, tt ))
        en_lsn_loss = loss_fn['mse']( en_diff_dist, Variable( tt.FloatTensor( en_diff_dist.size()).fill_(0) , requires_grad = False) )

        l2_diff_dist = torch.mean( torch.pow(output_l2[1][0] - output_l2[1][1], 2), 2).view(-1, args.num_dist)
        l2_logits = 1 / (l2_diff_dist + 1e-10)
        l2_lsn_acc = logit_to_acc(l2_logits, l2_batch[7]) * 100

        l2_diff_dist = torch.masked_select(l2_diff_dist, idx_to_emb( l2_batch[7].cpu().data.numpy(), args.num_dist, tt ))
        l2_lsn_loss = loss_fn['mse']( l2_diff_dist, Variable( tt.FloatTensor( l2_diff_dist.size()).fill_(0) , requires_grad = False) )

        final_loss += en_lsn_loss
        final_loss += l2_lsn_loss

    loss_dict["l1"]["lsn"]["loss"].update(en_lsn_loss.data[0])
    loss_dict["l1"]["lsn"]["acc"].update(en_lsn_acc)
    loss_dict["l2"]["lsn"]["loss"].update(l2_lsn_loss.data[0])
    loss_dict["l2"]["lsn"]["acc"].update(l2_lsn_acc)

    return final_loss

def forward_naka(images, labels, model, loss_dict, args, loss_fn, tt):
    final_loss = 0
    ####### ENCODER
    if args.train_enc_how == "two":
        langs = [args.src]
        lsns = [model.encoder_src]
    elif args.train_enc_how == "three":
        langs = [args.src, args.trg]
        lsns = [model.encoder_src, model.encoder_trg]

    for lang, lsn, kk in zip(langs, lsns, "enc_src enc_trg".split()):
        imgs, _, _, caps_mid, caps_mid_lens, _ = next_batch_naka_enc(images[lang], labels[lang], args.batch_size, tt)
        perm_idx = [ ( x + random.randint(1, args.batch_size) ) % args.batch_size for x in range(args.batch_size)]
        perm_idx = Variable( tt.LongTensor(np.array( perm_idx )) )

        h_img = model.beholder(imgs) # (batch_size, D_hid)
        rnn_out = lsn(caps_mid, caps_mid_lens) # (batch_size, D_hid)
        rnn_dist = rnn_out.index_select(0, perm_idx) # (batch_size, D_hid)

        if args.loss_type == "margin":
            dot_t = torch.sum( torch.mul( h_img, rnn_out ), 1).squeeze() # (batch_size)
            dot_d = torch.sum( torch.mul( h_img, rnn_dist ), 1).squeeze() # (batch_size)
            enc_loss = (args.margin - dot_t + dot_d).clamp(min=0.0)
            enc_loss = enc_loss.mean()

        elif args.loss_type == "reciprocal":
            rnn_total = torch.stack( (rnn_out, rnn_dist) ) # (2, batch_size, D_hid)
            h_img = torch.stack( (h_img, h_img) ) # (2, batch_size, D_hid)

            ans = torch.pow( h_img - rnn_total, 2)
            ans = ans.mean(2).squeeze().transpose(1,0) # (batch_size, 2)
            ans = 1 / (ans + 1e-8)

            which = Variable( tt.LongTensor( [0 for x in range(args.batch_size)] ) )
            for ii in range(args.batch_size):
                rr = np.random.rand(1)[0]
                if rr <= 0.5:
                    which[ii] = 1
                    ans[ii] = ans[ii].index_select(0, Variable(tt.LongTensor([1,0])))
            enc_loss = loss_fn(ans, which)

        loss_dict[kk].update(enc_loss.data[0] * args.lmbd)
        final_loss += enc_loss * args.lmbd

    ####### DECODER

    if args.train_dec_how == "img" or args.train_dec_how == "both":
        imgs, caps_in, caps_in_lens, _, _, caps_out = next_batch_naka_enc(images[args.trg], labels[args.trg], args.batch_size, tt)
        spk_h_img = model.beholder(imgs)
        spk_logits, _ = model.decoder_trg(spk_h_img, caps_in, caps_in_lens, "argmax")

        dec_loss = loss_fn(spk_logits, caps_out)
        loss_dict['dec_img'].update(dec_loss.data[0])
        final_loss += dec_loss

    if args.train_dec_how == "des" or args.train_dec_how == "both":
        _, caps_in, caps_in_lens, caps_mid, caps_mid_lens, caps_out = next_batch_naka_enc(images[args.trg], labels[args.trg], args.batch_size, tt)
        rnn_out = model.encoder_trg(caps_mid, caps_mid_lens) # (batch_size, D_hid)
        spk_logits, _ = model.decoder_trg(rnn_out, caps_in, caps_in_lens, "argmax")

        dec_loss = loss_fn(spk_logits, caps_out)
        loss_dict['dec_des'].update(dec_loss.data[0])
        final_loss += dec_loss

    return final_loss

