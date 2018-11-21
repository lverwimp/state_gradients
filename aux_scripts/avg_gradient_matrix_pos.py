#! /usr/bin/env python
# calculate average gradient for a certain delay
# and for a specific class
# if fifth argument is 'npz', then assume we have compressed numpy files

import sys, glob, os, re
import numpy as np
from utils import read_pos_tags

grad_dir = sys.argv[1]
delay = int(sys.argv[2])
pos_classes = sys.argv[3]
pos_f = sys.argv[4] # file containing the mapping from words to POS

if len(sys.argv) > 5:
    if sys.argv[5] != 'npz':
        raise ValueError("Third argument should be 'npz' if the gradient matrices have been compressed.")
        sys.exit(1)
    extension = 'npy'
else:
    extension = 'npz'

classes = [c.strip() for c in open(pos_classes).readlines()]
pos_mapping, _ = read_pos_tags(pos_f)

list_grads = glob.glob('{0}/timestep_*/delay_{1}/*.{2}'.format(grad_dir, delay, extension))

tmp_sum = 0.0
tmp_denom = 0.0
for grad_matrix in list_grads:

    mapping_f = os.path.join(os.path.dirname(grad_matrix), 'mapping.txt')
    words = []
    for line in open(mapping_f, 'r'):
        words.append(line.split()[1])

    for i, w in enumerate(words):
        if w in pos_mapping:
            pos_curr = pos_mapping[w]

            # check if the word is a class that we want to average over
            for p in pos_curr:
                if p in classes:
                    if extension == 'npy':
                        tmp_sum += np.load(grad_matrix)
                        tmp_denom += 1
                        break
                    else:
                        tmp_sum += np.load(grad_matrix)['arr_0'][i]
                        tmp_denom += 1
                        break
        else:
            print('{0} not in pos_mapping'.format(w))

avg = tmp_sum / tmp_denom

np.save(os.path.join(grad_dir, 'delay_{0}_avg_{1}.npy'.format(delay, os.path.basename(pos_classes))), avg)
