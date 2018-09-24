from __future__ import print_function
# import argparse
import torch
import torch.nn as nn
# # import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
from person_reid_nets import Siamese_Net
import resnet_custom
import torchvision.models as models
# from torchsummary import summary    # sigh not working on pyhon 2.7 Sep 2018
from loss_and_metrics import ContrastiveLoss, scores, plot_scores
from person_reid_nets import Siamese_Net
import msmt17_v1_utils
import math
import random
import torch.optim as optim
import os
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import argparse
import pickle
from IUV_stack_utils import *  #TODO


def combined_IUVstack_from_multiple_chips(dataload, pid, chip_paths, trainortest, combine_mode=None):
    # example:
    # combined_IUV_stack = chip_paths_to_combined_IUVstack(pid='0001', 
    #                                                      chip_paths=['/data/MSMT17_V1/train/0001/0001_011_01_0303morning_0036_1.jpg'
    #                                                   '/data/MSMT17_V1/train/0001/0001_051_01_0303morning_0758_0.jpg'
    #                                                   '/data/MSMT17_V1/train/0001/0001_045_01_0303morning_0755_1.jpg'
    #                                                   '/data/MSMT17_V1/train/0001/0001_067_14_0303morning_0981_1.jpg'
    #                                                   '/data/MSMT17_V1/train/0001/0001_018_07_0303morning_0033_1_ex.jpg'], 
    #                                                    trainortest = 'train')
    indiv_stacks = []
    for path in chip_paths:
        chipname = os.path.basename(path)
        im, IUV, INDS = dataload.get(trainortest=trainortest, 
                                     pid=pid, 
                                     chipname=chipname)
        if IUV is None or INDS is None:
            continue # skip processing this path
        ii = inds_value_of_most_prominent_person(INDS, mode='strict')
        if ii is None:   # strict mode "failed". Change mode.
            ii = inds_value_of_most_prominent_person(INDS, mode='most pixels')
        if ii is None:   # could be that INDS segmented no person(s).
            continue # skip processing this path
        new_IUV = IUV_with_only_this_inds_value(IUV, INDS, inds_value=ii)
        indiv_stacks.append(create_IUVstack(im, new_IUV))
    return combine_IUV_stacks(IUV_stack_list=indiv_stacks, mode=combine_mode)


def preprocess_IUV_stack(IUV_stack, device):
    # IUV_stack: Format is format of output of create_IUVstack(image_file, IUV_png_file).
    #            This code can be MODIFIED to also take in a HSV or HS (no V) type of IUVstack.
    # device: torch device. E.g. 'cuda' or 'cpu'.
    S = IUV_stack[:, 16:256-16, 16:256-16, :]    # Crop to 24x224x224x3
    S = normalize_to_reals_0to1(S)               # Normalize
    S = torch.Tensor(S)                          # Convert to torch tensors
    S = S.to(device)
    return S.view(24*3, 224, 224)


if __name__ == '__main__':
    
    # Injecting intersection suffiency check.   IUV utils > intersection check & n_elements_can_hold_info.  Basically redo if cannot find sufficient intersection.

    # argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--odir', type=str, default='tmp',
                        help='output dir. e.g. \'expt-1\'.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='gpu or cpu device. cuda cuda:0 cuda:1 cpu')
    parser.add_argument('--overfit', type=bool, default=True,
                        help='')                       
    args = parser.parse_args()
    
    val_samples = 3
    epochs = 5000
    epochs_between_saves = 20
    batch_size = 32 # take it as number of identities in a mini-batch. If 8 identities, there'll be 16 (8 x 2) pairs - 8 same pairs, 8 diff pairs. E.g. Jane-jane pair (same) , Jane-Tom pair (diff), Ben-Ben pair (same), Ben-Aaron pair (diff), ...
    plot_training_metric = True
    plot_loss = True
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    dataload = msmt17_v1_utils.MSMT17_V1_Load_Data_Utils(images_train_dir='/data/MSMT17_V1/train', 
                                                            images_test_dir='/data/MSMT17_V1/test', 
                                                            denseposeoutput_train_dir='/data/IUV-densepose/MSMT17_V1/train', 
                                                            denseposeoutput_test_dir='/data/IUV-densepose/MSMT17_V1/test')
    load_net_path = '' #'net-ep-2.chkpt'
    net = resnet_custom.resnet18(input_channels=24*3, num_classes=256)
    net = Siamese_Net(net)
    if load_net_path:
        net.load_state_dict(torch.load(load_net_path))
    net = net.to(device)
    contrastive_loss = ContrastiveLoss(option='two margin cosine', pos_margin=0.1, neg_margin=0.7)
    distance_type = 'cosine'
    # contrastive_loss = ContrastiveLoss(option='cosine')
    # distance_type = 'cosine'
    # contrastive_loss = ContrastiveLoss(option='euclidean')
    # distance_type = 'euclidean'
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    print(contrastive_loss.__dict__)

    optimizer = optim.Adam(net.parameters(), amsgrad=True, lr=0.0001) #defaults: lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0   https://pytorch.org/docs/stable/optim.html?highlight=optim#torch.optim.Adam

    # Make dir for this experiment:
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    else:
        print('Overwriting ', args.odir, '!'*100)
    
    train_loss_record = []
    val_loss_record = []
    most_info_in_an_input_so_far = 0.03 * (24 * 224 * 224 * 3) # 224 is h,w accepted into net. init.

    # Compute

    for epoch in range(epochs):
        if not args.overfit:
            pids_shuffled = np.random.permutation(dataload.train_persons_cnt)
        else:
            print('OVERFIT!' * 20)
            pids_shuffled = np.array([4, 8, 13])
            batch_size = 3
        #print(pids_shuffled.size)
        n_batches = int(math.ceil( pids_shuffled.size / (batch_size*1.0) ))
        # print(n_batches)
        train_loss_batches = []
        net.train()
        for bidx in range(n_batches):
            batch_pids = pids_shuffled.take(np.arange(batch_size) + batch_size * bidx, mode='wrap')
            batch_of_IUV_stacks = []
            for pid in batch_pids:
                #print('pid ', pid)
                pidstr = str(pid).zfill(4)
                n_chips_avail = dataload.num_chips('train', pidstr)
                chips = dataload.random_chips('train', pidstr, n_chips_avail)
                split1 = chips[ : int(len(chips)/2)]
                split2 = chips[int(len(chips)/2) : ]
                S1 = combined_IUVstack_from_multiple_chips(dataload, pid=pidstr, chip_paths=split1, trainortest='train', combine_mode='average v2')
                S2 = combined_IUVstack_from_multiple_chips(dataload, pid=pidstr, chip_paths=split2, trainortest='train', combine_mode='average v2')
                batch_of_IUV_stacks.append( (S1.copy(), S2.copy()) )
            #print('Done compute IUV stacks')
            input1s = []; input2s = []; targets = []
            for person in range(len(batch_of_IUV_stacks)):
                ######## Same person pair ########
                S1 = batch_of_IUV_stacks[person][0]
                S2 = batch_of_IUV_stacks[person][1]
                mask = get_intersection(S1, S2)          # intersect
                most_info_in_an_input_so_far = max(np.sum(mask), most_info_in_an_input_so_far) # update
                print('% IUV filled: ', 1.0 * np.sum(mask) / most_info_in_an_input_so_far)
                S1 = apply_mask_to_IUVstack(S1, mask)
                S2 = apply_mask_to_IUVstack(S2, mask)
                S1 = preprocess_IUV_stack(S1, device)
                S2 = preprocess_IUV_stack(S2, device)
                input1s.append(S1); input2s.append(S2); targets.append(1)
                ######## Diff persons pair ########
                persons = list(range(len(batch_of_IUV_stacks)))
                persons.remove(person)
                another_person = random.choice(persons)
                assert(person != another_person) # prevent edge cases like 1 person only in batch.
                S1 = batch_of_IUV_stacks[person][random.randint(0,1)]
                S2 = batch_of_IUV_stacks[another_person][random.randint(0,1)]
                mask = get_intersection(S1, S2)          # intersect
                most_info_in_an_input_so_far = max(np.sum(mask), most_info_in_an_input_so_far) # update
                print('% IUV filled: ', 1.0 * np.sum(mask) / most_info_in_an_input_so_far)
                S1 = apply_mask_to_IUVstack(S1, mask)
                S2 = apply_mask_to_IUVstack(S2, mask)
                S1 = preprocess_IUV_stack(S1, device)
                S2 = preprocess_IUV_stack(S2, device)
                input1s.append(S1); input2s.append(S2); targets.append(0)
            assert(len(input1s) > 0)
            assert( len(input1s) == len(input2s) and len(input2s) == len(targets) )
            input1s, input2s, targets = torch.stack(input1s), torch.stack(input2s), torch.Tensor(targets)
            #print(input1s.shape, input2s.shape, targets.shape)
            # input1s = torch.randn(4,24*3,224,224)                   #HACK just to get it to compile.
            # input2s = torch.randn(4,24*3,224,224)
            # targets = torch.Tensor([1,0,1,0])
            input1s, input2s, targets = input1s.to(device), input2s.to(device), targets.to(device) # yes! must re-assign
            optimizer.zero_grad()
            output1s, output2s = net(input1s, input2s)                       
            loss = contrastive_loss(output1s, output2s, targets)
            loss.backward()
            optimizer.step()

            train_loss_batches.append(loss.cpu().numpy().copy())

            if plot_training_metric:
                embeds1 = output1s[targets > 0.5,:].cpu().numpy().copy()
                embeds2 = output2s[targets > 0.5,:].cpu().numpy().copy()
                _, geniune_scores, imposter_scores = scores(embeds1, embeds2, distance_type)
                print('G', geniune_scores)
                print('IMP', imposter_scores)
                fig, plt = plot_scores(geniune_scores, imposter_scores, 'Scores', bin_width=0.05)
                fig.savefig(os.path.join(args.odir, 'tr-scores.jpg'))
                plt.close(fig)
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (bidx+1) * batch_size, pids_shuffled.size,
                100. * (bidx+1) / n_batches, loss.item()))
        
        train_loss_record.append( np.sum(train_loss_batches) / (1.0 * len(train_loss_batches)) )

        if plot_loss:
            fig = plt.figure()
            ax = fig.gca()
            plt.title('train & val loss')
            plt.plot(range(1, len(train_loss_record) + 1), train_loss_record, 'b')
            plt.ylabel('loss'); plt.xlabel('epochs')
            fig.savefig(os.path.join(args.odir, 'loss.jpg'))
            plt.close(fig)
        
        if (epoch + 1) % epochs_between_saves == 0:
            torch.save(net.state_dict(), os.path.join(args.odir, 'net-ep-{}.chkpt'.format(epoch+1)) )

        # for k epochs, run 1 validation.
        
