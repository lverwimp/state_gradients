#! /usr/bin/env python
# normalize embedding space

import sys, glob, time, os
import numpy as np
from decimal import *

emb_f = sys.argv[1]

# set this to True if you want to print the sanity checks
SANITY_CHECK = True

# calculate mean of all embeddings
all_emb = np.load(emb_f)
mean_emb = np.sum(all_emb, axis=0) / all_emb.shape[0]

# save mean embedding to file
mean_f = emb_f.replace('.npy','') + '_mean.npy'
np.save(mean_f, mean_emb)

# sanity check: check if mean of all normalized embeddings is 0 (or at least very small)
if SANITY_CHECK:
    all_norm_emb = all_emb - mean_emb
    mean_norm_emb = np.sum(all_norm_emb, axis=0) / all_norm_emb.shape[0]
    print('mean of mean-normalized embeddings (should be very small):')
    print(mean_norm_emb)

# calculate covariance
cov = np.cov(np.transpose(all_emb))

# save covariance matrix to file
covar_f = emb_f.replace('.npy','') + '_covar.npy'
np.save(covar_f, cov)

# calculate cholesky decomposition of covariance matrix
D = np.linalg.cholesky(cov)

# D = weights that will be used to normalize gradients
D_f = emb_f.replace('.npy','') + '_D_norm_grad.npy'
np.save(D_f, D)

# inverse of D = Z = weights that will be used to normalize the embeddings
Z = Dinv = np.linalg.inv(D)

Z_f = emb_f.replace('.npy','') + '_Z_norm_emb.npy'
np.save(Z_f, Z)

if SANITY_CHECK:
    # sanity check: Z * D * D^T * Z^T should be equal to identity
    print('Z * D * D.T * Z.T (should be equal to I):')
    print(Z * D * D.T * Z.T)

    # sanity check: condition number of Z (= SV_max / SV_min) should not be too large
    # singular values of D: SV_max / SV_min should not be larger than 100
    _, sZ, _ = np.linalg.svd(Z)
    print('condition number of Z (should be < 100): {0}'.format(sZ[0] / sZ[sZ.shape[0]-1]))

    # sanity check: variance of Z * embeddings should be 1
    norm_emb = np.zeros((all_emb.shape[0], all_emb.shape[1]))
    for i in range(all_emb.shape[0]):
        norm_emb[i] = np.matmul(Z, all_emb[i])
    cov = np.cov(np.transpose(norm_emb))
    print('new covariance matrix (should be close to I):')
    print(cov)
