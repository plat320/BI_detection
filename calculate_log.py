## Measure the detection performance - Kibok Lee
from __future__ import print_function
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_curve(dir_name, stypes = ['Baseline', 'Gaussian_LDA']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known = np.loadtxt('{}/confidence_{}_In.txt'.format(dir_name, stype), delimiter='\n')
        novel = np.loadtxt('{}/confidence_{}_Out.txt'.format(dir_name, stype), delimiter='\n')
        known.sort()                                                # sorting softmax results
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])                # maximum value of known and novel
        start = np.min([np.min(known),np.min(novel)])               # minimum value of known and novel
        num_k = known.shape[0]                                      # # of known
        num_n = novel.shape[0]                                      # # of novel
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)            # make np list size like total # of data
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)            # "
        tp[stype][0], fp[stype][0] = num_k, num_n                   # tp[stype][0] means # of data (initial value)
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:                             # 제일 작은것들끼리 비교했을 때 novel > known이면 FP -1,
                    n += 1
                    tp[stype][l+1] = tp[stype][l]                   # sustain true positive
                    fp[stype][l+1] = fp[stype][l] - 1               # -1 false positive
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1               # -1 true positive
                    fp[stype][l+1] = fp[stype][l]                   #  sustain false positive
        tpr95_pos = np.abs(tp[stype] / num_k - .80).argmin()        # true positive # / total known # (finding true positive 95% location)
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n     # 1 - false negative

    return tp, fp, tnr_at_tpr95

def metric(dir_name, save_fig, temp=1, mag=0, stypes = ['Bas', 'Gau'], verbose=True):
    tp, fp, tnr_at_tpr95 = get_curve(dir_name, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if stype=="PoT" and save_fig:
            plt.plot(fp[stype]/fp[stype][0], tp[stype]/tp[stype][0])
            plt.savefig(os.path.join(save_fig, "temp_{:.4f}_mag_{:.4f}.png".format(temp, mag)))
            plt.clf()

        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = -np.trapz(1.-fpr, tpr)
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = .5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
            print('')
    
    return results