#! /usr/bin/env python
# calculate
# - top 5 largest SVs
# - sum of all singular values
# if third argument is given, calculate gradients for average over specific class

import sys, glob
import numpy as np

grad_dir = sys.argv[1]
norm_weights = sys.argv[2] # output of normalize_equal_variance.py: <emb_f>_D_norm_grad.npy

D = np.load(norm_weights)

if len(sys.argv) > 3:
    pos = sys.argv[3]
    list_grads = glob.glob('{0}/delay_*_avg_{1}.npy'.format(grad_dir, pos))
else:
    list_grads = glob.glob('{0}/delay_*_avg.npy'.format(grad_dir))

for f in list_grads:
    delay = f.replace(grad_dir, '').replace('_avg.npy', '').replace('delay_', '')
    grad_matrix = np.load(f)
    norm_grad_matrix = np.matmul(grad_matrix, D)

    # calculate svd
    u, s, vh = np.linalg.svd(norm_grad_matrix)

	print('delay: {0}\tSV_1: {1}\tSV_2: {2}\tSV_3: {3}\tSV_4: {4}\tSV_5: {5}\sum: {6}'.format(
			delay, s[0], s[1], s[2], s[3], s[4], np.sum(s)))
