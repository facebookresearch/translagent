# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import torch
import cPickle as pkl
import codecs
import numpy as np
import collections

from torch.autograd import Variable

class Logger(object):
    def __init__(self, path, no_write=False, no_terminal=False):
        self.no_write = no_write
        if self.no_write:
            print "Don't write to file"
        else:
            self.log = codecs.open(path+"log.log", "wb", encoding="utf8")

        self.no_terminal = no_terminal
        self.terminal = sys.stdout

    def write(self, message):
        if not self.no_write:
            self.log.write(message)
        if not self.no_terminal:
            self.terminal.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def scr_path():
    return "/misc/kcgscratch1/ChoGroup/jason/translagent_public/"

def saved_results_path():
    return "/misc/kcgscratch1/ChoGroup/jason/saved_results/"

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pick(i1, i2, whichs):
    res = []
    img = [i1, i2]
    for idx, which in enumerate(whichs):
        res.append(img[which][idx])
    return res

def idx_to_onehot(indices, nb_digits): # input numpy array
    y = torch.LongTensor(indices).view(-1, 1)
    y_onehot = torch.FloatTensor(indices.shape[0], nb_digits)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot

def max_logit_to_onehot(logits):
    max_element, max_idx = torch.max(logits.cuda(), 1, keepdim=True)
    onehot = torch.FloatTensor(logits.size())
    onehot.zero_()
    onehot.scatter_(1, max_idx.data.long().cpu(), 1)
    onehot = Variable(torch.FloatTensor(onehot), requires_grad=False).cuda()
    return onehot, max_idx.data

def sample_logit_to_onehot(logits):
    #idx = torch.multinomial(logits, 1, replacement=True)
    indices = torch.multinomial(logits, 1)
    onehot = torch.FloatTensor(logits.size())
    onehot.zero_()
    for ii, jj in enumerate(indices.data.cpu().numpy().flatten().tolist()):
        onehot[ii][jj] = 1
    onehot = Variable(onehot, requires_grad=False).cuda()
    return onehot, indices.data

def logit_to_acc(logits, y): # logits: [batch_size, num_of_classes]
    y_max, y_max_idx = torch.max(logits, 1) # [batch_size]
    eq = torch.eq(y_max_idx, y)
    acc = float(eq.sum().data[0]) / float(eq.nelement())
    return acc

def logit_to_top_k(logits, y, k): # logits: [batch_size, num_of_classes]
    logits_sorted, indices = torch.sort(logits, 1, descending=True)
    y = y.view(-1, 1)
    indices = indices[:,:k]
    y_big = y.expand(indices.size())
    eq = torch.eq(indices, y_big)
    eq2 = torch.sum(eq, 1)
    #acc = float(eq2.sum().data[0]) / float(eq2.nelement())
    #return acc
    return eq2.sum().data[0], eq2.nelement()

def loss_and_acc(logits, labels, loss_fn):
    loss = loss_fn(logits, labels)
    acc = logit_to_acc(logits, labels)
    return (loss, acc)

def get_loss_dict():
    total_loss = {"agent1":{\
                        "spk":{\
                               "loss":0,\
                               "acc":0 },\
                        "lsn":{\
                               "loss":0,\
                               "acc":0 }\
                       },\
                  "agent2":{\
                        "spk":{\
                               "loss":0,\
                               "acc":0},\
                        "lsn":{\
                               "loss":0,\
                               "acc":0}\
                       }\
                 }
    return total_loss

def get_log_loss_dict():
    total_loss = {"agent1":{\
                        "spk":{\
                               "loss":AverageMeter(),\
                               "acc":AverageMeter() },\
                        "lsn":{\
                               "loss":AverageMeter(),\
                               "acc":AverageMeter() }\
                       },\
                  "agent2":{\
                        "spk":{\
                               "loss":AverageMeter(),\
                               "acc":AverageMeter()},\
                        "lsn":{\
                               "loss":AverageMeter(),\
                               "acc":AverageMeter()\
                              }\
                       }\
                 }
    return total_loss

def get_avg_from_loss_dict(log_loss_dict):
    res = get_loss_dict()
    for k1, v1 in log_loss_dict.iteritems(): # agent1 / agent2
        for k2, v2 in v1.iteritems(): # spk / lsn
            for k3, v3 in v2.iteritems(): # loss / acc
                res[k1][k2][k3] = v3.avg
    return res

#print "epoch {:5d} train | alpha {:.3f} | snd {:.4f} {:.2f}% | rcv {:.4f} {:.2f}% ".format(epoch, args.alpha, snd_losses.avg, snd_accs.avg*100, rcv_losses.avg, rcv_accs.avg*100)
def print_loss(epoch, alpha, avg_loss_dict, mode="train"):
    prt_msg = "epoch {:5d} {} | alpha {:.3f} ".format(epoch, mode, alpha)
    #avg_loss_dict = get_avg_from_loss_dict(loss_dict)
    #for k1, v1 in avg_loss_dict.iteritems():
    for agent in "agent1 agent2".split():
        prt_msg += "| {} :".format(agent) # agent1 / agent2
        #for k2, v2 in v1.iteritems():
        for person in "spk lsn".split():
            prt_msg += " {}".format(person) # spk / lsn
            #for k3, v3 in v2.iteritems(): # loss / acc
            prt_msg += " {:.3f}".format(avg_loss_dict[agent][person]["loss"])
            prt_msg += " {:.2f}".format(avg_loss_dict[agent][person]["acc"])
            prt_msg += "% |"
    return prt_msg

def bergsma_words(lang):
    words = open('/misc/kcgscratch1/ChoGroup/jason/fair/bergsma_data/Share/Lexicons/500/pictureWords.{}'.format(lang)).readlines()
    words = [x.strip() for x in words if x.strip() != ""]
    words = [x.split("\t") for x in words]
    words = [words[key][1] for key in range(len(words))]
    return words

def recur_mkdir(dir):
    ll = dir.split("/")
    ll = [x for x in ll if x != ""]
    for idx in range(len(ll)):
        ss = "/".join(ll[0:idx+1])
        check_mkdir("/"+ss)

def check_mkdir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)
