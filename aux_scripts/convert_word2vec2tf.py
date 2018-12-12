#! /usr/bin/env python
# convert word2vec embeddings to format used by TF scripts

import sys, os
import numpy as np

word2vec_f = sys.argv[1]
dict_tf = sys.argv[2]
new_word2vec_dir = sys.argv[3]
new_dict_tf = sys.argv[4]

out_dict = open(new_dict_tf, 'w')
vocab = [line.split()[0] for line in open(dict_tf).readlines()]

idx_new_dict = 0
list_emb = []
for line in open(word2vec_f, 'r'):
    l = line.strip().split(' ')
    if len(l) > 2:
        if l[0] in vocab:
            # write to new dict file
            out_dict.write('{0}\t{1}\n'.format(l[0], idx_new_dict))
            
            # convert to numpy array
            float_l = [float(el) for el in l[1:]]
            emb = np.array(float_l)
            
            list_emb.append(emb)

            # increase idx_new_dict
            idx_new_dict += 1

# stack all vectors into a matrix and save it
all_emb = np.stack(list_emb)
np.save(os.path.join(new_word2vec_dir, 'all.npy'), all_emb)

out_dict.close()
