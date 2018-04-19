# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import sys
import cPickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util import *

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

class TwoAgents(torch.nn.Module):
    def __init__(self, args):
        super(TwoAgents, self).__init__()
        self.agent1 = Agent(args.l1, args.l2, args)
        self.agent2 = Agent(args.l2, args.l1, args)

        self.agents = [self.agent1, self.agent2]
        self.num_cat = args.num_cat
        self.no_share_bhd = args.no_share_bhd
        self.train_how = args.train_how
        self.D_img = args.D_img
        self.D_hid = args.D_hid
        self.l1 = args.l1
        self.l2 = args.l2

    def forward(self, data1, data2):
        a_spk_img, b_lsn_imgs = data1 # spk_imgs : (batch_size, 2048)
        b_spk_img, a_lsn_imgs = data2 # lsn_imgs : (batch_size, num_dist, 2048)
        spk_inputs = [a_spk_img, b_spk_img] # [a, b]
        spk_outputs = [] # [a, b] logits
        lsn_inputs = [a_lsn_imgs, b_lsn_imgs] # [a, b]
        lsn_outputs = [] # [a, b]
        comm_onehots = [] # [a, b]
        comm_actions = []
        num_dist = b_lsn_imgs.size()[1]

        ##### Speaker #####
        for agent, spk_img in zip(self.agents, spk_inputs): # [a, b]
            spk_h_img = spk_img
            spk_h_img = agent.beholder1(spk_h_img) if self.no_share_bhd else agent.beholder(spk_h_img)

            spk_logits, comm_onehot, comm_action = agent.speaker(spk_h_img)
            spk_outputs.append(spk_logits)
            # spk_logits : (batch_size, num_cat)
            # comm_onehot : (batch_size, num_cat)
            # comm_action : (batch_size)

            comm_onehots.append(comm_onehot.cuda())
            comm_actions.append(comm_action.cuda())

        comm_onehots = comm_onehots[::-1] # [b, a]

        ##### Listener #####
        for agent, comm_onehot, lsn_imgs in zip(self.agents, comm_onehots, lsn_inputs):
            # lsn_imgs : (batch_size, num_dist, 2048)
            lsn_imgs = lsn_imgs.view(-1, self.D_img) # (batch_size * num_dist, D_img)
            lsn_h_imgs = agent.beholder2(lsn_imgs) if self.no_share_bhd else agent.beholder(lsn_imgs)
            lsn_h_imgs = lsn_h_imgs.view(-1, num_dist, self.D_hid) # (batch_size, num_dist, D_hig)

            lsn_dot = agent.listener(lsn_h_imgs, comm_onehot) # (batch_size, num_dist)
            lsn_outputs.append(lsn_dot)

        return (spk_outputs[0], lsn_outputs[1]), (spk_outputs[1], lsn_outputs[0]), comm_actions

    def translate_from_en(self, sample=False, print_neighbours=False):
        l1_dic = get_idx_to_cat(args.l1)
        l2_dic = get_idx_to_cat(args.l2)
        assert (len(l1_dic) == self.num_cat) and (len(l2_dic) == self.num_cat)

        result = { 1:{1:[],0:[]}, 0:{1:[],0:[]}  }

        batch_size = 640
        #keys = np.random.randint(0, self.num_cat, size=batch_size)
        keys = np.arange(0, self.num_cat)
        labels = torch.LongTensor(keys).view(batch_size, 1)

        onehot1 = torch.FloatTensor(batch_size, self.num_cat)
        onehot1.zero_()
        onehot1.scatter_(1, labels, 1)
        onehot1 = Variable(onehot1, requires_grad=False).cuda()

        if sample:
            logits1 = self.agent2.translate(onehot1)
            onehot2, idx2 = sample_logit_to_onehot(logits1)

            logits2 = self.agent1.translate(onehot2)
            onehot3, idx3 = sample_logit_to_onehot(logits2)

        else:
            logits1 = self.agent2.translate(onehot1)
            onehot2, idx2 = max_logit_to_onehot(logits1)

            logits2 = self.agent1.translate(onehot2)
            onehot3, idx3 = max_logit_to_onehot(logits2)

        _, indices1 = torch.sort(logits1, 1, descending=True)
        _, indices2 = torch.sort(logits2, 1, descending=True)
        indices1 = indices1.cpu().data.numpy()
        indices2 = indices2.cpu().data.numpy()

        for idx in xrange(labels.nelement()):
            #print u"{:>25} -> {:>25} -> {:>25}".format(l1_dic[labels[idx][0]], l2_dic[idx2[idx][0]], l1_dic[idx3[idx][0]])
            if print_neighbours:
                for k in range(5):
                    print u"{:>25} -> {:>25}".format(l1_dic[labels[idx][0]], l2_dic[indices1[idx][k]])
                for k in range(5):
                    print u"{:>25}    {:>25} -> {:>25}".format("", l2_dic[idx2[idx][0]], l1_dic[indices2[idx][k]])

            if labels[idx][0] == idx2[idx][0]:
                right1 =1
            else:
                right1= 0

            if labels[idx][0] == idx3[idx][0]:
                right2 =1
            else:
                right2= 0
            result[right1][right2].append( (l1_dic[labels[idx][0]], l2_dic[idx2[idx][0]], l1_dic[idx3[idx][0]]) )
            if right2 == 0 or right1 == 0:
                print (right1, right2)
                for k in range(5):
                    print u"{:>25} -> {:>25}".format(l1_dic[labels[idx][0]], l2_dic[indices1[idx][k]])
                for k in range(5):
                    print u"{:>25}    {:>25} -> {:>25}".format("", l2_dic[idx2[idx][0]], l1_dic[indices2[idx][k]])
        return result

    def en2de(self, onehot):
        logits = self.agent2.translate(onehot)
        return logits

    def de2en(self, onehot):
        logits = self.agent1.translate(onehot)
        return logits

    def en2de2en(self, onehot):
        logits1 = self.agent2.translate(onehot)
        onehot2, _ = max_logit_to_onehot(logits1)
        logits2 = self.agent1.translate(onehot2)
        return logits2

    def precision(self, keys, bs):
        ks = [1, 5, 20]
        result = []
        rounds = ["{}->{} (agent2/{}) ".format(self.l1, self.l2, self.l2),
                  "{}->{} (agent1/{}) ".format(self.l2, self.l1, self.l1),
                  "{}->{}->{} (agent2/{}->agent1/{}) ".format(self.l1, self.l2, self.l1, self.l2, self.l1)]

        for which_round, round_ in enumerate(rounds):
            acc = [[0,0] for x in range(len(ks))]
            cnt = 0
            for batch_idx in xrange(int(math.ceil( float(len(keys)) / bs ) ) ):
                labels_ = np.arange(batch_idx * bs , min(len(keys), (batch_idx+1) * bs ) )
                labels_ = keys[labels_]
                batch_size = len(labels_)
                cnt += batch_size

                labels = torch.LongTensor(labels_).view(-1)
                labels = torch.unsqueeze(labels, 1)
                labels = Variable(labels, requires_grad=False).cuda()

                onehot = torch.FloatTensor(batch_size, self.num_cat)
                onehot.zero_()
                onehot.scatter_(1, labels.data.cpu(), 1)
                onehot = Variable(onehot, requires_grad=False).cuda()

                if which_round == 0:
                    logits = self.en2de(onehot)
                elif which_round == 1:
                    logits = self.de2en(onehot)
                elif which_round == 2:
                    logits = self.en2de2en(onehot)

                for prec_idx, k in enumerate(ks):
                    right, total = logit_to_top_k(logits, labels, k)
                    acc[prec_idx][0] += right
                    acc[prec_idx][1] += total
            assert( cnt == len(keys) )
            assert( acc[0][1] == len(keys) )

            pm = round_
            for prec_idx, k in enumerate(ks):
                curr_acc = float(acc[prec_idx][0]) / acc[prec_idx][1] * 100
                result.append( curr_acc )
                pm += "| P@{} {:.2f}% ".format(k, curr_acc )
            print pm

        return result

class Agent(torch.nn.Module):
    def __init__(self, native, foreign, args):
        super(Agent, self).__init__()
        if args.no_share_bhd:
            print "Not sharing visual system for each agent."
            self.beholder1 = Beholder(args.D_img, args.D_hid, args.dropout)
            self.beholder2 = Beholder(args.D_img, args.D_hid, args.dropout)
        else:
            print "Sharing visual system for each agent."
            self.beholder = Beholder(args.D_img, args.D_hid, args.dropout)

        self.speaker = Speaker(native, foreign, args.D_hid, args.num_cat, args.dropout, args.temp, args.hard, args.tt)
        self.listener = Listener(native, foreign, args.D_hid, args.num_cat, args.dropout)

    def forward():
        return

    def translate(self, comm_onehot):
        lsn_emb_msg = self.listener.emb(comm_onehot) # (batch_size, D_hid)
        spk_logits = self.speaker.hid_to_cat(lsn_emb_msg) # (batch_size, num_cat)
        return spk_logits

class Beholder(torch.nn.Module):
    def __init__(self, D_img, D_hid, dropout):
        super(Beholder, self).__init__()
        self.img_to_hid = torch.nn.Linear(D_img, D_hid) # shared visual system
        self.drop = torch.nn.Dropout(p=dropout)

    def forward(self, img):
        img = self.drop(img)
        h_img = self.img_to_hid(img)
        return h_img

class Speaker(torch.nn.Module):
    def __init__(self, native, foreign, D_hid, num_cat, dropout, temp, hard, tt):
        super(Speaker, self).__init__()
        self.hid_to_cat = torch.nn.Linear(D_hid, num_cat, bias=False) # Speaker
        self.drop = torch.nn.Dropout(p=dropout)
        self.num_cat = num_cat
        self.temp = temp
        self.hard = hard
        self.tt = tt
        self.native, self.foreign = native, foreign

    def forward(self, h_img):
        #h_img = self.drop(h_img)
        spk_logits = self.hid_to_cat(h_img) # (batch_size, num_cat)
        comm_onehot, comm_label = gumbel_softmax(spk_logits, temp=self.temp, hard=self.hard, tt=self.tt) # (batch_size, num_cat)
        logits_grad = spk_logits.grad
        output_grad = comm_onehot.grad
        return spk_logits, comm_onehot, comm_label

    def nn_words(self, batch_size = 5):
        word_idx = np.random.randint(0, self.num_cat, size=batch_size)
        for idx in word_idx:
            self.compute_dot_for_all(idx)

    def compute_dot_for_all(self, idx):
        l1_dic = get_idx_to_cat(self.native)
        assert len(l1_dic) == self.num_cat

        emb = torch.FloatTensor(self.hid_to_cat.weight.data.cpu()) # [num_cat, D_hid]
        vec = emb[idx] # [1, D_hid]

        vec_exp = torch.unsqueeze(vec,0).expand(emb.size()) # (num_cat, D_hid)
        prod = torch.mul(vec_exp, emb) # [num_cat, D_hid]
        prod = torch.sum(prod, 1) # [num_cat]
        norm1 = torch.norm(vec_exp, 2, 1) # [num_cat]
        norm2 = torch.norm(emb, 2, 1) # (num_cat)
        norm = torch.mul(norm1, norm2) # [num_cat]

        ans = prod / norm
        ans = ans.view(self.num_cat)

        logits_sorted, indices = torch.sort(ans, dim=0, descending=True)
        indices = indices[:5].cpu().numpy()

        print u"{} -> {}".format(l1_dic[idx].decode('utf8'), u", ".join([u"{} ({})".format(l1_dic[idx1].decode('utf-8'), "{:0.2f}".format(ans[idx1])) for idx1 in indices]))

class Listener(torch.nn.Module):
    def __init__(self, native, foreign, D_hid, num_cat, dropout):
        super(Listener, self).__init__()
        self.emb = torch.nn.Linear(num_cat, D_hid, bias=False)

        self.D_hid = D_hid
        self.num_cat = num_cat
        self.native, self.foreign = native, foreign

    def forward(self, lsn_h_imgs, comm_onehot):
        # lsn_h_imgs : (batch_size, num_dist, D_hid)
        # comm_onehot : (batch_size, num_cat)
        num_dist = lsn_h_imgs.size()[1]
        lsn_hid_msg = self.emb(comm_onehot) # (batch_size, D_hid)

        lsn_hid_msg = lsn_hid_msg.unsqueeze(1).repeat(1, num_dist, 1)
        lsn_hid_msg = lsn_hid_msg.view(-1, num_dist, self.D_hid) # (batch_size, num_dist, D_hid)

        diff = torch.pow( lsn_hid_msg - lsn_h_imgs, 2)
        diff = torch.mean( diff, 2 ) # (batch_size, num_dist)
        diff = 1 / (diff + 1e-10)
        diff = diff.squeeze()

        return diff # (batch_size, num_dist)

    def nn_words(self, batch_size = 5):
        word_idx = np.random.randint(0, self.num_cat, size=batch_size)
        for idx in word_idx:
            self.compute_dot_for_all(idx)

    def compute_dot_for_all(self, idx):
        l1_dic = bergsma_words(self.foreign)
        #assert len(l1_dic) == self.num_cat

        emb = torch.FloatTensor(torch.t(self.emb.weight.data.cpu())) # [num_cat, D_hid]
        vec = emb[idx] # [1, D_hid]

        vec_exp = torch.unsqueeze(vec,0).expand(emb.size()) # (num_cat, D_hid)
        prod = torch.mul(vec_exp, emb) # [num_cat, D_hid]
        prod = torch.sum(prod, 1) # [num_cat]
        norm1 = torch.norm(vec_exp, 2, 1) # [num_cat]
        norm2 = torch.norm(emb, 2, 1) # (num_cat)
        norm = torch.mul(norm1, norm2) # [num_cat]

        ans = prod / norm
        ans = ans.view(-1)

        logits_sorted, indices = torch.sort(ans, dim=0, descending=True)
        indices = indices[:5].cpu().numpy()

        print u"{} -> {}".format(l1_dic[idx].decode('utf-8'), u", ".join([u"{} ({})".format(l1_dic[idx1].decode('utf-8'), "{:0.2f}".format(ans[idx1])) for idx1 in indices]))

