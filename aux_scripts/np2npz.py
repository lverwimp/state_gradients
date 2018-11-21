#! /usr/bin/env python
# compress all numpy files in the input directory

import sys, subprocess, os
import numpy as np

dir_np = sys.argv[1]

os.chdir(dir_np)
ls_output = subprocess.check_output(['ls *.npy| sort -n'.format(dir_np)], shell=True)
list_arrays = ls_output.split('\n')

tmp_l = []

for f in list_arrays:
    if f != '':
        tmp_l.append(np.load(f))

np.savez_compressed(os.path.join(dir_np, 'all.npz'), tmp_l)
