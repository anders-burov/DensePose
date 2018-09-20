from __future__ import print_function
import numpy as np
import os
import argparse
# import pickle
import msmt17_v1_utils
from IUV_stack_utils import *  #TODO
from example_basic_train import combined_IUVstack_from_multiple_chips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--odir', type=str, default='/data/IUV-densepose/MSMT17_V1/precomputed',
                        help='output dir')
    parser.add_argument('--n_sets', type=int, default=10,
                        help='Number of sets of precomputed test people to generate.')
    parser.add_argument('--start_at', type=int, default=0,
                        help='Set id to start at. This is to help make this script more run-able in parallel.')
    args = parser.parse_args()
    [print(arg, ':', getattr(args, arg)) for arg in vars(args)]
    dataload = msmt17_v1_utils.MSMT17_V1_Load_Data_Utils(images_train_dir='/data/MSMT17_V1/train', 
                                                            images_test_dir='/data/MSMT17_V1/test', 
                                                            denseposeoutput_train_dir='/data/IUV-densepose/MSMT17_V1/train', 
                                                            denseposeoutput_test_dir='/data/IUV-densepose/MSMT17_V1/test')
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    for setid in range(args.start_at, args.n_sets):
        setdir = os.path.join(args.odir, str(setid))
        if not os.path.exists(setdir):
            os.makedirs(setdir)
        for pid in range(3060):
            outfile = os.path.join(setdir, '{}.npz'.format(pid))
            if os.path.exists(outfile):
                continue
            print('Working on set id', setid, 'pid', pid)
            pidstr = str(pid).zfill(4)
            n_chips_avail = dataload.num_chips('test', pidstr)
            chips = dataload.random_chips('test', pidstr, n_chips_avail)
            split1 = chips[ : int(len(chips)/2)]
            split2 = chips[int(len(chips)/2) : ]
            S1 = combined_IUVstack_from_multiple_chips(dataload, pid=pidstr, chip_paths=split1, trainortest='test', combine_mode='average v2')
            S2 = combined_IUVstack_from_multiple_chips(dataload, pid=pidstr, chip_paths=split2, trainortest='test', combine_mode='average v2')
            np.savez_compressed(outfile, 
                                S1=S1.astype(np.float16), 
                                S2=S2.astype(np.float16))