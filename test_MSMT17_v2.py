from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from person_reid_nets import Siamese_Net
import torchvision.models as models
# from torchsummary import summary    # sigh not working on pyhon 2.7 Sep 2018
from loss_and_metrics import ContrastiveLoss, scores, plot_scores
import math
import random
import torch.optim as optim
import torchvision.models as models
import os
import argparse
import pickle
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from person_reid_nets import Siamese_Net
import resnet_custom
import msmt17_v1_utils
from IUV_stack_utils import *  #TODO
from cmc import count as cmc_count




if __name__ == '__main__':
    script_start_time = "{:%m-%d-%H-%M-%S}".format(datetime.now()); print('script_start_time ', script_start_time)
    parser = argparse.ArgumentParser()
    parser.add_argument('--odir', type=str, default='tmp',
                        help='output dir. e.g. \'expt-1\'.')
    parser.add_argument('--idir', type=str, default='/data/IUV-densepose/MSMT17_V1/precomputed',
                        help='input dir.')
    parser.add_argument('--setid', type=int, default=0,
                        help='precomputed test set id')
    parser.add_argument('--device', type=str, default='cuda:3',
                        help='gpu or cpu device. cuda cuda:0 cuda:1 cpu')
    parser.add_argument('--n_probes', type=int, default=3,
                        help='num of probes')
    parser.add_argument('--n_gallery', type=int, default=3,
                        help='num of people in gallery')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='To prevent OOM error on device running neural netowrk.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num workers on dataloader.')
    parser.add_argument('--net', type=str, default=None,
                        help='path to network. e.g. \"expt-32-twomarg-0.1-0.7/net-ep-20.chkpt\"')
    parser.add_argument('--contrastlossopt', type=str, default="two margin cosine",
                        help='e.g. \"two margin cosine\"')
    parser.add_argument('--pos_margin', type=float, default=None,
                        help='e.g. 0.1')
    parser.add_argument('--neg_margin', type=float, default=None,
                        help='e.g. 0.7')
    parser.add_argument('--distance_type', type=str, default='cosine',
                        help='e.g. cosine')
    args = parser.parse_args()
    [print(arg, ':', getattr(args, arg)) for arg in vars(args)]
    assert(args.n_gallery >= args.n_probes)
    # assert(args.n_gallery % args.batch_size == 0)
    loss = ContrastiveLoss(option=args.contrastlossopt, pos_margin=args.pos_margin, neg_margin=args.neg_margin)
    print("[NOTE]  Ensure loss used is similar to loss used during training if you\'re comparing training loss to test/val loss.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    dataload = msmt17_v1_utils.MSMT17_V1_Load_Data_Utils(images_train_dir='/data/MSMT17_V1/train', 
                                                            images_test_dir='/data/MSMT17_V1/test', 
                                                            denseposeoutput_train_dir='/data/IUV-densepose/MSMT17_V1/train', 
                                                            denseposeoutput_test_dir='/data/IUV-densepose/MSMT17_V1/test')
    # Make dir for this experiment:
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    logbk = {} # A dict for recording stuff computed.
    # Print loss:
    print(loss.__dict__)
    # Load net:
    net = resnet_custom.resnet18(input_channels=24*3, num_classes=256)
    net = Siamese_Net(net)
    net.load_state_dict(torch.load(args.net))
    net = net.to(device)
    net.eval()
    # Get samples:
    pids_shuffled = np.random.permutation(dataload.test_persons_cnt)
    glabels = pids_shuffled[:args.n_gallery]
    plabels = glabels[:args.n_probes]
    logbk['glabels'] = glabels
    logbk['plabels'] = plabels
    similarity_mat = np.zeros((len(glabels), len(plabels)))
    loss_mat = None # np.zeros_like(similarity_mat)  # TODO. loss needs to have reduce=False option.
    intersection_mat = np.zeros_like(similarity_mat)
    dataset_test = msmt17_v1_utils.Dataset_Test(precomputed_test_path=args.idir, 
                                setid=args.setid, 
                                glabels=glabels, 
                                plabels=plabels)
    dataloader = torch.utils.data.DataLoader(dataset_test,
                                            batch_size=args.batch_size, 
                                            shuffle=False, # so can map to similarity matrix more easily (i guess?)
                                            num_workers=args.num_workers,
                                            drop_last = False)
    with torch.no_grad():
        for i_batch, data_batched in enumerate(dataloader):
            print('{} Batch [{}/ {}]'.format("{:%m-%d-%H-%M-%S}".format(datetime.now()), 
                                                        i_batch + 1, 
                                                        len(dataloader)))
            S_gals, S_probes, targets, g_pids, p_pids, g_picks, p_picks, row_idxs, col_idxs, intersection_amts, flattened_idxs = data_batched
            #print(flattened_idxs, S_gals.shape, targets.shape, row_idxs, col_idxs)
            S_probes, S_gals, targets = S_probes.to(device), S_gals.to(device), targets.to(device)
            embs_p, embs_g = net(S_probes, S_gals)
            #loss_ = loss(emb1s, emb2s, targets).cpu().numpy().copy()  # TODO add reduce so that two_margin_cosine_loss can do this.
            # loss_mat[start_idx:end_idx, col] = loss_
            # print(loss_mat[0:16,0:16])
            score_mat, _, _ = scores(embs_g, embs_p, args.distance_type)
            relevant_scores = np.diagonal(score_mat)  # off diagonal elems are irrelevant!
            # for i in range(embs_g.shape[0]):
            #     score, _, _ = scores(embs_g[[i],:], embs_p[[i],:], args.distance_type)
            #     print(score)
            similarity_mat[row_idxs, col_idxs] = relevant_scores
            # print(similarity_mat)
            intersection_mat[row_idxs, col_idxs] = intersection_amts
            # print(intersection_mat)
    logbk['similarity_mat'] = similarity_mat
    logbk['intersection_mat'] = intersection_mat
    pickle.dump(logbk, open(os.path.join(args.odir, 'test-logbk-{}.pkl'.format(script_start_time)), 'wb'), protocol=2)
    # Plot CMC:
    distmat = 1 - logbk['similarity_mat']
    cmc_values = cmc_count(distmat=distmat, glabels=logbk['glabels'], plabels=logbk['plabels'], n_selected_labels=None, n_repeat=1)
    print(cmc_values)
    fig = plt.figure()
    ax = fig.gca()
    plt.title('CMC')
    plt.plot(range(1, len(cmc_values) + 1), cmc_values, 'b')
    plt.ylabel('Match rate'); plt.xlabel('Rank')
    #plt.show()
    fig.savefig(os.path.join(args.odir, 'test-cmc-{}.jpg'.format(script_start_time)))
    plt.close(fig)
    np.set_printoptions(precision=4)
    print(similarity_mat)
    np.set_printoptions() #reset
    print(intersection_mat)

