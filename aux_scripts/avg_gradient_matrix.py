#! /usr/bin/env python
# calculate average gradient for a certain delay

import sys, glob, os, re
import numpy as np

grad_dir = sys.argv[1]
delay = int(sys.argv[2])

list_grads = glob.glob('{0}/timestep_*/delay_{1}/*.npy'.format(grad_dir, delay))

tmp_sum = 0.0
tmp_denom = 0.0
for grad_matrix in list_grads:
    tmp_sum += np.load(grad_matrix)
    tmp_denom += 1

avg = tmp_sum / tmp_denom

np.save(os.path.join(grad_dir, 'delay_{0}_avg.npy'.format(delay)), avg)
