# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import operator
import math
import sys
import os
import cPickle as pkl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util import *

millis = int(round(time.time() * 1000))
torch.manual_seed(millis)
torch.cuda.manual_seed(millis)

def sample_gumbel(shape, tt=torch, eps=1e-20):
    U = Variable(tt.FloatTensor(shape).uniform_(0, 1))
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temp, tt=torch):
    y = ( logits + sample_gumbel(logits.size(), tt) ) / temp
    return F.softmax(y)

def gumbel_softmax(logits, temp, hard, tt=torch):
    y = gumbel_softmax_sample(logits, temp, tt) # (batch_size, num_cat)
    y_max, y_max_idx = torch.max(y, 1, keepdim=True)
    if hard:
        y_hard = tt.FloatTensor(y.size()).zero_().scatter_(1, y_max_idx.data, 1)
        y = Variable( y_hard - y.data, requires_grad=False ) + y

    return y, y_max_idx

class NakaAgent(torch.nn.Module):
    def __init__(self, native, foreign, args):
        super(NakaAgent, self).__init__()
        if args.no_share_bhd:
            print "Not sharing visual system for each agent."
            self.beholder1 = Beholder(args)
            self.beholder2 = Beholder(args)
        else:
            print "Sharing visual system for each agent."
            self.beholder  = Beholder(args)

        self.decoder_trg = Speaker(native, args)
        self.encoder_src = RnnListener(foreign, args)
        if args.train_enc_how == "three":
            self.encoder_trg = RnnListener(native, args)

        self.tt = torch if args.cpu else torch.cuda
        self.native, self.foreign = native, foreign
        self.i2w, self.w2i = args.i2w, args.w2i
        self.unit_norm = args.unit_norm

        self.D_hid = args.D_hid

        self.beam_width = args.beam_width
        self.norm_pow = args.norm_pow

    def forward(self):
        return

    def translate(self, src, decode_how="beam"):
        # src : (batch_size, seq_len)
        batch_size = len(src)
        src_lens = [len(src_i) for src_i in src]
        seq_len = max(src_lens)

        src = np.array( [ np.lib.pad( src_i, (0, seq_len - len(src_i) ), 'constant', constant_values=(0,0) ) for src_i in src ] )
        src = Variable( self.tt.LongTensor( src ), requires_grad=False ) # (batch_size, seq_len)

        concept = self.encoder_src(src, src_lens) # (batch_size, D_hid)
        if self.unit_norm:
            norm = torch.norm(concept, 2, 1, keepdim=True).expand_as(concept).detach() + 1e-9
            concept = concept / norm

        if decode_how == "beam":
            gen_idx = self.decoder_trg.beam_search(concept, self.beam_width, self.norm_pow)
        elif decode_how == "greedy":
            gen_idx = self.decoder_trg.sample(concept, True)

        trg = decode(gen_idx, self.i2w[self.native])
        return trg

class NMT(torch.nn.Module):
    def __init__(self, src, trg, args):
        # vocab_size, num_layers, num_directions, i2w, w2i : 2 separate arguments
        super(NMT, self).__init__()

        self.decoder = Speaker(trg, args)
        self.encoder = RnnListener(src, args)

        self.tt = torch if args.cpu else torch.cuda
        self.trg, self.src = trg, src
        self.i2w, self.w2i = args.i2w, args.w2i

        self.D_hid = args.D_hid

        self.beam_width = args.beam_width
        self.norm_pow = args.norm_pow

    def forward(self, src_caps_in, src_caps_in_lens, trg_sorted_idx, trg_caps_in, trg_caps_in_lens):
        enc_hid = self.encoder(src_caps_in, src_caps_in_lens) # (batch_size, D_hid)
        enc_hid = torch.index_select(enc_hid, 0, trg_sorted_idx)
        dec_logits, _ = self.decoder(enc_hid, trg_caps_in, trg_caps_in_lens, "argmax")
        return dec_logits

    def translate(self, src, args):
        # src : (batch_size, seq_len)
        batch_size = len(src)
        src_lens = [len(src_i) for src_i in src]
        seq_len = max(src_lens)

        src = np.array( [ np.lib.pad( src_i, (0, seq_len - len(src_i) ), 'constant', constant_values=(0,0) ) for src_i in src ] )
        src = Variable( self.tt.LongTensor( src ), requires_grad=False ) # (batch_size, seq_len)

        concept = self.encoder(src, src_lens) # (batch_size, D_hid)

        if args.decode_how == "beam":
            gen_idx = self.decoder.beam_search(concept, args.beam_width, args.norm_pow)
        elif args.decode_how == "greedy":
            gen_idx = self.decoder.sample(concept, True)

        trg = decode(gen_idx, self.i2w[self.trg])
        return trg

class TwoAgents(torch.nn.Module):
    def __init__(self, args):
        super(TwoAgents, self).__init__()
        self.en_agent = Agent('en', args.l2, args)
        self.l2_agent = Agent(args.l2, 'en', args)

        self.agents = [self.en_agent, self.l2_agent]
        self.no_share_bhd = args.no_share_bhd
        self.D_img = args.D_img
        self.D_hid = args.D_hid

    def forward(self, data1, data2, spk_sample_how):
        a_spk_img, b_lsn_imgs, a_spk_caps_in, a_spk_cap_lens = data1 # spk_imgs : (batch_size, 2048)
        b_spk_img, a_lsn_imgs, b_spk_caps_in, b_spk_cap_lens = data2 # lsn_imgs : (batch_size, num_dist, 2048)
        spk_inputs = [a_spk_img, b_spk_img] # [a, b]
        spk_caps_ins = [a_spk_caps_in, b_spk_caps_in] # [a, b]
        spk_cap_lens = [a_spk_cap_lens, b_spk_cap_lens] # [a, b]
        spk_outputs = [] # [a, b] logits

        lsn_inputs = [a_lsn_imgs, b_lsn_imgs] # [a, b]
        lsn_outputs = [] # [a, b]

        comm_actions = []
        num_dist = b_lsn_imgs.size()[1]

        ##### Speaker #####
        for agent, spk_input, spk_cap_in, spk_cap_len in zip(self.agents, spk_inputs, spk_caps_ins, spk_cap_lens): # [a, b]
            if self.no_share_bhd:
                spk_h_img = agent.beholder1(spk_input) # shared
            else:
                spk_h_img = agent.beholder(spk_input) # shared

            spk_logits, comm_action = agent.speaker(spk_h_img, spk_cap_in, spk_cap_len, spk_sample_how) # NOTE argmax / gumbel
            spk_outputs.append(spk_logits)
            comm_actions.append(comm_action)

        comm_actions = comm_actions[::-1] # [b, a]

        ##### Listener #####
        for agent, lsn_imgs, comm, spk_cap_len in zip(self.agents, lsn_inputs, comm_actions, spk_cap_lens):
            lsn_imgs = lsn_imgs.view(-1, self.D_img)
            if self.no_share_bhd:
                lsn_h_imgs = agent.beholder2(lsn_imgs)
            else:
                lsn_h_imgs = agent.beholder(lsn_imgs)
            lsn_h_imgs = lsn_h_imgs.view(-1, num_dist, self.D_hid)

            rnn_hid = agent.listener(comm[:,:-1], spk_cap_len-1)
            rnn_hid = rnn_hid.unsqueeze(1).repeat(1, num_dist, 1) # (batch_size, num_dist, D_hid)

            lsn_outputs.append( (rnn_hid, lsn_h_imgs) )

        return (spk_outputs[0], lsn_outputs[1]), (spk_outputs[1], lsn_outputs[0]), comm_actions

    def en2l2(self, src):
        trg = self.en_agent.translate(src)
        return trg

    def l22en(self, src):
        trg = self.l2_agent.translate(src)
        return trg

class Agent(torch.nn.Module):
    def __init__(self, native, foreign, args):
        # vocab_size, num_layers, num_directions, i2w, w2i : 2 separate arguments
        super(Agent, self).__init__()
        if args.no_share_bhd:
            print "Not sharing visual system for each agent."
            self.beholder1 = Beholder(args)
            self.beholder2 = Beholder(args)
        else:
            print "Sharing visual system for each agent."
            self.beholder  = Beholder(args)

        self.speaker = Speaker(native, args)
        self.listener = RnnListener(foreign, args)
        self.tt = torch if args.cpu else torch.cuda
        self.native, self.foreign = native, foreign
        self.i2w, self.w2i = args.i2w, args.w2i
        self.unit_norm = args.unit_norm

        self.beam_width = args.beam_width
        self.norm_pow = args.norm_pow

        self.D_hid = args.D_hid

    def forward(self, src_caps_in, src_caps_in_lens, trg_sorted_idx, trg_caps_in, trg_caps_in_lens):
        enc_hid = self.listener(src_caps_in, src_caps_in_lens) # (batch_size, D_hid)
        enc_hid = torch.index_select(enc_hid, 0, trg_sorted_idx)
        dec_logits, _ = self.speaker(enc_hid, trg_caps_in, trg_caps_in_lens, "gumbel")
        return dec_logits

    def translate(self, src, decode_how="beam"):
        # src : (batch_size, seq_len)
        batch_size = len(src)
        src_lens = [len(src_i) for src_i in src]
        seq_len = max(src_lens)

        src = np.array( [ np.lib.pad( src_i, (0, seq_len - len(src_i) ), 'constant', constant_values=(0,0) ) for src_i in src ] )
        src = Variable( self.tt.LongTensor( src ), requires_grad=False ) # (batch_size, seq_len)

        concept = self.listener(src, src_lens) # (batch_size, D_hid)
        if self.unit_norm:
            norm = torch.norm(concept, 2, 1, keepdim=True).expand_as(concept).detach() + 1e-9
            concept = concept / norm

        if decode_how == "beam":
            gen_idx = self.speaker.beam_search(concept, self.beam_width, self.norm_pow)
        elif decode_how == "greedy":
            gen_idx = self.speaker.sample(concept, True)

        trg = decode(gen_idx, self.i2w[self.native])
        return trg

class Beholder(torch.nn.Module):
    def __init__(self, args):
        super(Beholder, self).__init__()
        self.img_to_hid = torch.nn.Linear(args.D_img, args.D_hid) # shared visual system
        self.unit_norm = args.unit_norm
        self.drop = nn.Dropout(p=args.dropout)
        self.two_fc = args.two_fc
        if self.two_fc:
            self.hid_to_hid = torch.nn.Linear(args.D_hid, args.D_hid)

    def forward(self, img):
        h_img = img

        h_img = self.img_to_hid(h_img)

        h_img = self.drop(h_img)

        if self.two_fc:
            h_img = self.hid_to_hid( F.relu( h_img ) )

        if self.unit_norm:
            #norm = torch.norm(h_img, p=2, dim=1) + 1e-9
            norm = torch.norm(h_img, p=2, dim=1, keepdim=True).detach() + 1e-9
            h_img = h_img / norm.expand_as(h_img)
        return h_img

class RnnListener(torch.nn.Module):
    def __init__(self, lang, args):
        super(RnnListener, self).__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, args.num_layers['lsn'][lang], batch_first=True) if args.num_directions['lsn'][lang] == 1 else \
                   nn.GRU(args.D_emb, args.D_hid, args.num_layers['lsn'][lang], batch_first=True, bidirectional=True)
        self.emb = nn.Embedding(args.vocab_size[lang], args.D_emb, padding_idx=0)
        self.hid_to_hid = nn.Linear(args.num_directions['lsn'][lang] * args.D_hid, args.D_hid)
        self.drop = nn.Dropout(p=args.dropout)

        self.D_hid = args.D_hid
        self.D_emb = args.D_emb
        self.num_layers = args.num_layers['lsn'][lang]
        self.num_directions = args.num_directions['lsn'][lang]
        self.vocab_size = args.vocab_size[lang]
        self.i2w = args.i2w[lang]
        self.w2i = args.w2i[lang]
        self.unit_norm = args.unit_norm

        self.tt = torch if args.cpu else torch.cuda

    def forward(self, spk_msg, spk_msg_lens):
        # spk_msg : (batch_size, seq_len)
        # spk_msg_lens : (batch_size)
        batch_size = spk_msg.size()[0]
        seq_len = spk_msg.size()[1]

        h_0 = Variable( self.tt.FloatTensor(self.num_layers * self.num_directions, batch_size, self.D_hid).zero_() )
        spk_msg_emb = self.emb(spk_msg) # (batch_size, seq_length, D_emb)
        spk_msg_emb = self.drop(spk_msg_emb)

        pack = torch.nn.utils.rnn.pack_padded_sequence(spk_msg_emb, spk_msg_lens, batch_first=True)

        # input (batch_size, seq_len, D_emb)
        # h_0 (num_layers * num_directions, batch_size, D_hid)
        _, h_n = self.rnn(pack, h_0)
        # output (batch_size, seq_len, D_hid * num_directions)
        # h_n (num_layers * num_directions, batch_size, D_hid)

        h_n = h_n[-self.num_directions:,:,:]
        out = h_n.transpose(0,1).contiguous().view(batch_size, self.num_directions * self.D_hid)
        # out (batch_size, num_layers * num_directions * D_hid)
        out = self.hid_to_hid( out )
        #out = self.hid_to_hid( out )
        # out (batch_size, D_hid)

        if self.unit_norm:
            #norm = torch.norm(out, p=2, dim=1) + 1e-9
            norm = torch.norm(out, p=2, dim=1, keepdim=True).detach() + 1e-9
            out = out / norm.expand_as(out)

        return out

class Speaker(torch.nn.Module):
    def __init__(self, lang, args):
        super(Speaker, self).__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, args.num_layers['spk'][lang], batch_first=True)
        self.emb = nn.Embedding(args.vocab_size[lang], args.D_emb, padding_idx=0)
        self.hid_to_voc = nn.Linear(args.D_hid, args.vocab_size[lang])

        self.D_emb = args.D_emb
        self.D_hid = args.D_hid
        self.num_layers = args.num_layers['spk'][lang]
        self.drop = nn.Dropout(p=args.dropout)

        self.vocab_size = args.vocab_size[lang]
        self.i2w = args.i2w[lang]
        self.w2i = args.w2i[lang]

        self.temp = args.temp
        self.hard = args.hard
        self.tt = torch if args.cpu else torch.cuda
        self.seq_len = args.seq_len[lang]

    def forward(self, h_img, caps_in, caps_in_lens, sample_how):
        # h_img : (batch_size, D_hid)
        # caps_in : (batch_size, seq_len)
        # caps_in_lens : (batch_size)
        batch_size = caps_in.size()[0]
        seq_len = caps_in.size()[1]

        h_img = h_img.view(1, batch_size, self.D_hid).repeat(self.num_layers, 1, 1)
        caps_in_emb = self.emb(caps_in) # (batch_size, seq_length, D_emb)
        caps_in_emb = self.drop(caps_in_emb)

        pack = torch.nn.utils.rnn.pack_padded_sequence(caps_in_emb, caps_in_lens, batch_first=True)

        # input (batch_size, seq_len, D_emb)
        # h0 (num_layers, batch_size, D_hid)
        output, _ = self.rnn(pack, h_img)
        # output (batch_size, seq_len, D_hid)
        # hn (num_layers, batch_size, D_hid)
        output_data = output.data
        logits = self.hid_to_voc( output_data )

        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # unpacked (batch_size, seq_len, D_hid)
        unpacked_logits = self.hid_to_voc( unpacked.contiguous().view(-1, self.D_hid) )
        # unpacked (batch_size * seq_len, vocab_size)

        if sample_how == "argmax":
            _, comm_label = torch.max(unpacked_logits, 1)
        elif sample_how == "gumbel":
            _, comm_label = gumbel_softmax(unpacked_logits, self.temp, self.hard, self.tt) # (batch_size, num_cat)

        comm_label = comm_label.view(batch_size, seq_len)
        len_scat = [range(x) + [0] * (seq_len - x) for x in caps_in_lens]
        mask = Variable( self.tt.LongTensor(batch_size, seq_len).zero_().scatter_(1, self.tt.LongTensor(len_scat), 1) )
        comm_label = comm_label * mask

        return logits, comm_label

    def sample(self, h_img, argmax):
        batch_size = h_img.size()[0]
        start = [self.w2i[ "<BOS>" ] for ii in range(batch_size)]
        gen_idx = [[] for ii in range(batch_size)]
        done = np.array( [False for ii in range(batch_size)] )

        # h_img : (batch_size, D_hid)
        h_img = h_img.unsqueeze(0).view(1, -1, self.D_hid).repeat(self.num_layers, 1, 1)
        # hid : (num_layers, batch_size, D_hid)
        hid = h_img
        ft = self.tt.LongTensor(start).view(-1).unsqueeze(1) # (batch_size, 1)
        input = self.emb( Variable( ft ) ) # (batch_size, D_emb)

        for idx in range(self.seq_len):
            output, hid = self.rnn( input, hid )

            output = output.view(-1, self.D_hid)
            output = self.hid_to_voc( output )
            output = output.view(-1, self.vocab_size)

            if argmax:
                topv, topi = output.data.topk(1)
                # topi (batch_size, 1)
                top1 = topi.view(-1).cpu().numpy()
            else:
                top1 = torch.multinomial(output, 1).cpu().data.numpy()

            for ii in range(batch_size):
                if self.i2w[ top1[ii] ] == "<EOS>":
                    done[ii] = True
                if not done[ii]:
                    gen_idx[ii].append( top1[ii] )

            if np.array_equal( done, np.array( [True for x in range(batch_size) ] ) ):
                break

            input = self.emb( Variable( topi ) )

        return gen_idx

    def beam_search(self, h_ctx, width, norm_pow): # # out (batch_size, D_hid)
        voc_size, batch_size = self.vocab_size, h_ctx.size()[0]
        live = [ [ ( 0.0, [ self.w2i[ "<BOS>" ] ], 0 ) ] for ii in range(batch_size) ]
        dead = [ [] for ii in range(batch_size) ]
        num_dead = [0 for ii in range(batch_size)]

        ft = self.tt.LongTensor( [ self.w2i[ "<BOS>" ] for ii in range(batch_size) ] )[:,None]
        input = self.emb( Variable( ft ) ) # (batch_size, 1, D_emb)

        #hid = h_ctx.view(batch_size, self.num_layers, self.D_hid).transpose(0,1)\
        #        .contiguous()
        hid = h_ctx[None,:,:].repeat(self.num_layers, 1, 1) # hid : (num_layers, batch_size, D_hid)

        for tidx in range(self.seq_len):
            # input (batch_size * width, 1, D_emb)
            # h_0 (num_layers, batch_size * width, D_hid)
            output, hid = self.rnn( input, hid )
            # output (batch_size * width, 1, D_hid)
            # h_n (num_layers, batch_size * width, D_hid)

            cur_prob = F.log_softmax( self.hid_to_voc( output.view(-1, self.D_hid) ))\
                    .view(batch_size, -1, voc_size).data # (batch_size, width, vocab_size)
            pre_prob = self.tt.FloatTensor( [ [ x[0] for x in ee ] for ee in live ] ).view(batch_size, -1, 1) \
                    # (batch_size, width, 1)
            total_prob = cur_prob + pre_prob # (batch_size, width, voc_size)
            total_prob = total_prob.view(batch_size, -1)

            _, topi_s = total_prob.topk( width, dim=1)
            topv_s = cur_prob.view(batch_size, -1).gather(1, topi_s)
            # (batch_size, width)

            new_live = [ [] for ii in range(batch_size) ]
            for bidx in range(batch_size):
                num_live = width - num_dead[bidx]
                if num_live > 0:
                    tis = topi_s[bidx][:num_live]
                    tvs = topv_s[bidx][:num_live]
                    for eidx, (topi, topv) in enumerate(zip(tis, tvs)): # NOTE max width times
                        if topi % voc_size == self.w2i[ "<EOS>" ] :
                            dead[bidx].append( (  live[bidx][ topi / voc_size ][0] + topv, \
                                                  live[bidx][ topi / voc_size ][1] + [ topi % voc_size ],\
                                                  topi) )
                            num_dead[bidx] += 1
                        else:
                            new_live[bidx].append( (    live[bidx][ topi / voc_size ][0] + topv, \
                                                        live[bidx][ topi / voc_size ][1] + [ topi % voc_size ],\
                                                        topi) )
                while len(new_live[bidx]) < width:
                    new_live[bidx].append( (    -99999999, \
                                                [0],\
                                                0) )
            live = new_live

            if num_dead == [width for ii in range(batch_size)]:
                break

            in_vocab_idx = [ [ x[2] % voc_size for x in ee ] for ee in live ] # NOTE batch_size first
            input = self.emb( Variable( self.tt.LongTensor( in_vocab_idx ) ).view(-1) ).view(-1, 1, self.D_emb) \
                    # input (batch_size * width, 1, D_emb)

            bb = 1 if tidx == 0 else width
            in_width_idx = [ [ x[2] / voc_size + bbidx * bb for x in ee ] for bbidx, ee in enumerate(live) ] \
                    # live : (batch_size, width)
            hid = hid.index_select( 1, Variable( self.tt.LongTensor( in_width_idx ).view(-1) ) ).\
                    view(self.num_layers, -1, self.D_hid)
            # h_0 (num_layers, batch_size * width, D_hid)

        for bidx in range(batch_size):
            if num_dead[bidx] < width:
                for didx in range( width - num_dead[bidx] ):
                    (a, b, c) = live[bidx][didx]
                    dead[bidx].append( (a, b, c)  )

        #dead_ = [ [ ( float(a) / ( float( math.pow(5+len(b), norm_pow) ) / math.pow(5+1, norm_pow) ) , b, c) for (a,b,c) in ee] for ee in dead]
        dead_ = [ [ ( float(a) / math.pow(len(b), norm_pow) , b, c) for (a,b,c) in ee] for ee in dead]
        ans = []
        for dd_ in dead_:
            dd = sorted( dd_, key=operator.itemgetter(0), reverse=True )
            ans.append( dd[0][1][1:-1] )
        return ans

