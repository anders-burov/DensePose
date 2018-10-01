from __future__ import print_function
# import math
# import random
import os
import argparse
import pickle
import torch
# import torch.nn as nn
import numpy as np
from person_reid_nets import Siamese_Net
# import torchvision.models as models
# from torchsummary import summary    # sigh not working on pyhon 2.7 Sep 2018
from loss_and_metrics import ContrastiveLoss, scores #, plot_scores
# import torch.optim as optim
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
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
    parser.add_argument('--device', type=str, default='cuda:0',
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
                        help='path to network architecture. e.g. \"expt-32-twomarg-0.1-0.7/net-09-20-14-31-52.pt\"')
    parser.add_argument('--chkpt', type=str, default=None,
                        help='path to network\'s checkpoint. e.g. \"expt-32-twomarg-0.1-0.7/net-ep-20.chkpt\"')
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
    print('Actual device: ', device)
    dataload = msmt17_v1_utils.MSMT17_V1_Load_Data_Utils(images_train_dir='/data/MSMT17_V1/train', 
                                                            images_test_dir='/data/MSMT17_V1/test', 
                                                            denseposeoutput_train_dir='/data/IUV-densepose/MSMT17_V1/train', 
                                                            denseposeoutput_test_dir='/data/IUV-densepose/MSMT17_V1/test')
    # Make dir for this experiment:
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    logbk = {} # A dict for recording stuff computed.
    
    print(loss.__dict__)

    # Load net:
    # resnet_args = {'block':resnet_custom.BasicBlock, 
    #                'layers':[2, 2, 2, 2], 
    #                'inplanes':256,
    #                'planes_of_layers':(256, 512, 1024, 2048),
    #                'input_channels':24*3,
    #                'num_classes':256}
    # print('resnet_args:', resnet_args)
    # net = resnet_custom.ResNet(**resnet_args)
    if args.net is not None:
        net = torch.load(args.net)
    else:
        net = resnet_custom.resnet18(input_channels=24*3, num_classes=256)
    net.load_state_dict(torch.load(args.chkpt))
    net = Siamese_Net(net)   # TODO can remove eventually as siamese-izing a net is for training only.
    net = net.to(device)
    net.eval()  # IMPT #
    
    pids_allowed = list(range(dataload.test_persons_cnt))
    # pids_allowed = [0,1,2,3] # For testing.
    similarity_mat = np.zeros((len(pids_allowed), len(pids_allowed)))

    dataset = msmt17_v1_utils.Dataset_msmt17_testprecomputed(precomputed_path=args.idir, 
                                                                    setid=args.setid, 
                                                                    pids_allowed=pids_allowed, 
                                                                    dataload=dataload)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=False, # so can map to similarity matrix more easily (i guess?)
                                            num_workers=args.num_workers,
                                            drop_last = False)
    all_embs_1, all_embs_2 = [], []
    with torch.no_grad():
        for i_batch, data_batched in enumerate(dataloader):
            print('{} Batch [{}/ {}]'.format("{:%m-%d-%H-%M-%S}".format(datetime.now()), 
                                                        i_batch + 1, 
                                                        len(dataloader)))
            S1s, S2s = data_batched
            S1s, S2s = S1s.to(device), S2s.to(device)
            # print(S1s.shape)
            embs_1, embs_2 = net(S1s, S2s)
            all_embs_1.append(embs_1)
            all_embs_2.append(embs_2)
    all_embs_1 = np.concatenate(all_embs_1)
    all_embs_2 = np.concatenate(all_embs_2)
    score_mat, genuine_scores, imposter_scores = scores(all_embs_1, all_embs_2, args.distance_type)
    logbk['similarity_mat'] = score_mat
    logbk['genuine_scores'] = genuine_scores
    logbk['imposter_scores'] = imposter_scores
    pickle.dump(logbk, open(os.path.join(args.odir, 'test-logbk-{}.pkl'.format(script_start_time)), 'wb'), protocol=2)
    # Plot CMC:
    distmat = 1 - logbk['similarity_mat']
    cmc_values = cmc_count(distmat=distmat, glabels=pids_allowed, plabels=pids_allowed, n_selected_labels=None, n_repeat=1)
    try:
        print('CMC values ranked 1 to 80:', cmc_values[:80])
        print('Rank 1,5,10,20:', cmc_values[[0,4,9,19]])
    except:
        print('CMC:', cmc_values)
    fig = plt.figure()
    ax = fig.gca()
    plt.title('CMC')
    plt.plot(range(1, len(cmc_values) + 1), cmc_values, 'b')
    plt.ylabel('Match rate'); plt.xlabel('Rank')
    #plt.show()
    fig.savefig(os.path.join(args.odir, 'test-cmc-{}.jpg'.format(script_start_time)))
    plt.close(fig)
    # np.set_printoptions(precision=4)
    # print(similarity_mat)
    # np.set_printoptions() #reset
    # print(intersection_mat)