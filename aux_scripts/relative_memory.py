#! /usr/bin/env python
# calculate relative memory =
# norm(avg gradient matrix * normalized difference vector) / largest singular value for avg gradient matrix

import sys, glob, os
import numpy as np

diff_vector = sys.argv[1]
grad_dir = sys.argv[2]
norm_emb_f = sys.argv[3]
if len(sys.argv) > 4:
    name_avg = sys.argv[4]
    list_grads = glob.glob('{0}/delay_*_avg_{1}_norm.npy'.format(grad_dir, name_avg))
else:
    list_grads = glob.glob('{0}/delay_*_avg_norm.npy'.format(grad_dir))

# load difference vector: difference between average embedding
# for 1 property and average embedding for another property
diff = np.load(diff_vector)
norm_emb = np.load(norm_emb_f)
norm_pca_diff = np.matmul(norm_emb, diff)
# normalize difference vector
norm_diff = norm_pca_diff / np.linalg.norm(norm_pca_diff)

for f in list_grads:
    delay = os.path.basename(f).replace('delay_','').replace('_avg.npy','')

    # calculate numerator: norm(avg gradient matrix * normalized difference vector)
    grad_matrix = np.load(f)
    mult = np.dot(grad_matrix, norm_diff)
    norm = np.linalg.norm(mult)

    # calculate svd
    u, s, vh = np.linalg.svd(grad_matrix)

    # divide by the largest singular value to get relative memory
    rel_mem = norm / s[0]

    print('delay: {0}\trelative memory: {1}'.format(delay, rel_mem))
