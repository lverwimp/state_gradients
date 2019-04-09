#! /usr/bin/env python
# calculate cosine distance between a difference vector
# and a subspace spanned by the v's corresponding to
# the X (indicated by thr) largest SVs

import sys, glob
import numpy as np

diff_vector = sys.argv[1]
grad_dir = sys.argv[2]
thr = int(sys.argv[3])
norm_emb_f = sys.argv[4]
norm_grad_f = sys.argv[5]
if len(sys.argv) > 6:
    name_avg = sys.argv[6]
    list_grads = glob.glob('{0}/delay_*_avg_{1}.npy'.format(grad_dir, name_avg))
else:
    list_grads = glob.glob('{0}/delay_*_avg.npy'.format(grad_dir))

norm_emb = np.load(norm_emb_f)
norm_grad = np.load(norm_grad_f)

# load difference vector: difference between average embedding
# for 1 property and average embedding for another property
diff = np.load(diff_vector)
norm_pca_diff = np.matmul(norm_emb, diff)
norm_diff = norm_pca_diff / np.linalg.norm(norm_pca_diff)

for f in list_grads:
    delay = f.replace(grad_dir, '').replace('_avg.npy', '').replace('delay_', '')

    grad_matrix = np.load(f)
    grad_matrix = np.matmul(grad_matrix, norm_grad)
	
    # calculate svd
    u, s, vh = np.linalg.svd(grad_matrix)

    vn = vh[:thr+1]

    sum_svs = np.sum(s)
    sum_slice_svs = np.sum(s[:thr+1])
    perc_sum = (sum_slice_svs / sum_svs)*100

    cos_dist = np.linalg.norm(np.dot(vn, norm_diff))

    print('delay: {0}\tcosine distance: {1}\tpercentage total sum: {2}'.format(delay, cos_dist, perc_sum))
