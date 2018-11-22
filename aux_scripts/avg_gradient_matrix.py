#! /usr/bin/env python
# calculate average gradient for a certain delay
# if third argument is 'npz', then assume we have compressed numpy files

import sys, glob, os, re
import numpy as np

grad_dir = sys.argv[1]
delay = int(sys.argv[2])

if len(sys.argv) > 3:
    if sys.argv[3] != 'npz':
        raise ValueError("Third argument should be 'npz' if the gradient matrices have been compressed.")
        sys.exit(1)
    extension = 'npz'
else:
    extension = 'npy'

list_grads = glob.glob('{0}/timestep_*/delay_{1}/*.{2}'.format(grad_dir, delay, extension))

tmp_sum = 0.0
tmp_denom = 0.0
for grad_matrix in list_grads:
    if extension == 'npy':
        tmp_sum += np.load(grad_matrix)
        tmp_denom += 1
    else:
        gr_m = np.load(grad_matrix)['arr_0']
        tmp_sum += np.sum(gr_m, axis=0)
        tmp_denom += gr_m.shape[0]

avg = tmp_sum / tmp_denom

np.save(os.path.join(grad_dir, 'delay_{0}_avg.npy'.format(delay)), avg)
