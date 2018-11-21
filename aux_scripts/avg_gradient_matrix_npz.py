#! /usr/bin/env python
# calculate average gradient for a certain delay
# can be used for compressed gradient matrices

import sys, glob, os, re
import numpy as np

grad_dir = sys.argv[1]
delay = int(sys.argv[2])

list_grads = glob.glob('{0}/timestep_*/delay_{1}/*.npz'.format(grad_dir, delay))

tmp_sum = 0.0
tmp_denom = 0.0
for grad_matrix in list_grads:
    gr_m += np.load(grad_matrix)['arr_0']
    tmp_sum += np.sum(gr_m, axis=0)
    tmp_denom += gr_m.shape[0]

avg = tmp_sum / tmp_denom

np.save(os.path.join(grad_dir, 'delay_{0}_avg.npy'.format(delay)), avg)
