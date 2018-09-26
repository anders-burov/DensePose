from __future__ import print_function
from datetime import datetime
import math
import random
import os
import argparse
import pickle
import numpy as np
from welford import Welford
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
# from torchsummary import summary    # sigh not working on pyhon 2.7 Sep 2018
import resnet_custom
from person_reid_nets import Siamese_Net
# from IUV_stack_utils import *  #TODO
import msmt17_v1_utils
from loss_and_metrics import ContrastiveLoss, scores, plot_scores
from cmc import count as cmc_count


if __name__ == '__main__':
    script_start_time = "{:%m-%d-%H-%M-%S}".format(datetime.now()); print('script_start_time ', script_start_time)
    
    # Injecting intersection suffiency check.   IUV utils > intersection check & n_elements_can_hold_info.  Basically redo if cannot find sufficient intersection.

    parser = argparse.ArgumentParser()
    parser.add_argument('--overfit', type=int, default=1,
                        help='')
    parser.add_argument('--odir', type=str, default='tmp',
                        help='output dir. e.g. \'expt-1\'.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='gpu or cpu device. cuda cuda:0 cuda:1 cpu')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train for.')
    parser.add_argument('--epochs_between_saves', type=int, default=10,
                        help='Save after this many epochs')
    parser.add_argument('--batch_size', type=int, default=50,
                        help="If batch size is 4, there\'ll be 4 positive pairs and 4 negative pairs per batch. Implies 8 pairs per batch.")
    parser.add_argument('--val_size', type=int, default=100,
                        help="Validation size.")                    
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num workers on dataloaders.')
    args = parser.parse_args()

    logbk = {}

    dataload = msmt17_v1_utils.MSMT17_V1_Load_Data_Utils(images_train_dir='/data/MSMT17_V1/train', 
                                                            images_test_dir='/data/MSMT17_V1/test', 
                                                            denseposeoutput_train_dir='/data/IUV-densepose/MSMT17_V1/train', 
                                                            denseposeoutput_test_dir='/data/IUV-densepose/MSMT17_V1/test')

    if args.overfit == 1:
        print('OVERFIT!' * 20)
        logbk['train_pids_allowed'] = tuple([4,8,12,13,14])
        logbk['val_pids_allowed'] = tuple([2001, 2002, 2003, 2004, 3001, 3002, 3003, 3004])
        args.batch_size = 3
        args.val_size = len(logbk['val_pids_allowed'])
    else:
        logbk['train_pids_allowed'] = tuple(range(dataload.train_persons_cnt))
        assert(len(logbk['train_pids_allowed']) > 10 and len(logbk['train_pids_allowed']) <= 1041)
        val_pids_shuffled = np.random.permutation(dataload.test_persons_cnt)
        logbk['val_pids_allowed'] = tuple(val_pids_shuffled[:args.val_size])
    
    print('val_pids_allowed:', logbk['val_pids_allowed'])
    [print(arg, ':', getattr(args, arg)) for arg in vars(args)]

    plot_training_metric = True
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Actual device: ', device)
    
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
    
    
    # Train data generator:
    train_set = msmt17_v1_utils.Dataset_msmt17(dataload=dataload, trainortest='train', pids_allowed=logbk['train_pids_allowed'], mask_inputs=False, combine_mode='average v2')
    train_generator = torch.utils.data.DataLoader(train_set,
                                            batch_size=args.batch_size, 
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            drop_last = False)
    # Validation data generator:
    validation_set = msmt17_v1_utils.Dataset_msmt17(dataload=dataload, trainortest='test', pids_allowed=logbk['val_pids_allowed'], mask_inputs=False, combine_mode='average v2')
    validation_generator = torch.utils.data.DataLoader(validation_set, 
                                            batch_size=args.batch_size, 
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            drop_last = False)

    
    logbk['train_losses'] = [] # each element is 1 epoch.
    logbk['validation_losses'] = [] # each element is 1 epoch.
    logbk['cmc'] = [] # each element is 1 epoch.
    logbk['intersection_amt_stats'] = Welford()
    logbk['intersection_amt_max'] = 0

    #most_info_in_an_input_so_far = 0.03 * (24 * 224 * 224 * 3) # 224 is h,w accepted into net. init.
    
    for epoch in range(args.epochs):
        train_loss_per_batch = []
        for i_batch, trn_data_batch in enumerate(train_generator):
            print('{} Batch [{}/ {}]'.format("{:%m-%d-%H-%M-%S}".format(datetime.now()), 
                                                        i_batch + 1, 
                                                        len(train_generator)))
            S_pos1s, S_pos2s, S_neg1s, S_neg2s, targets_pos, targets_neg, interx_amts_pos_pair, interx_amts_neg_pair, masks_pos_pair, masks_neg_pair, pos_pids, neg_pids = trn_data_batch
            
            # print(S_pos1s.shape, S_pos2s.shape, S_neg1s.shape)
            # print(targets_pos.shape, targets_pos[0:3])
            # print(targets_neg[0:3])
            # print(interx_amts_pos_pair[0:3])
            # print(masks_pos_pair.shape, masks_neg_pair.shape)
            # print(pos_pids)
            # print(neg_pids)
            
            # Update intersection amt stats:
            interx_amts_1 = interx_amts_pos_pair.view(-1).cpu().numpy().copy()
            interx_amts_2 = interx_amts_neg_pair.view(-1).cpu().numpy().copy()
            interx_amts_list = np.ndarray.tolist(np.concatenate((interx_amts_1, interx_amts_2)))
            logbk['intersection_amt_stats'](interx_amts_list)                                           #update
            logbk['intersection_amt_max'] = max(logbk['intersection_amt_max'], max(interx_amts_list))   #update
            print('Intersection', logbk['intersection_amt_stats'], 'max:', logbk['intersection_amt_max'])
            # print('% IUV filled <p>: ', 1.0 * interx_amt_pos_pair / self.intersection_amt_most_encountered)
            # print('% IUV filled <n>: ', 1.0 * interx_amt_neg_pair / self.intersection_amt_most_encountered)
            
            input1s, input2s, targets = torch.cat((S_pos1s, S_neg1s)), torch.cat((S_pos2s, S_neg2s)), torch.cat((targets_pos, targets_neg))
            # print(input1s.shape, input2s.shape, targets.shape)
            # print(targets)
            # input1s = torch.randn(4,24*3,224,224)                   #HACK just to get it to compile.
            # input2s = torch.randn(4,24*3,224,224)
            # targets = torch.Tensor([1,0,1,0])
            input1s, input2s, targets = input1s.to(device), input2s.to(device), targets.to(device) # yes! must re-assign
            net.train()         # IMPT #
            optimizer.zero_grad()
            output1s, output2s = net(input1s, input2s)                       
            loss = contrastive_loss(output1s, output2s, targets.float())
            loss.backward()
            optimizer.step()

            train_loss_per_batch.append(loss.item())

            if plot_training_metric and i_batch == 0:
                embeds1 = output1s[targets > 0.5,:].cpu().detach().numpy().copy()
                embeds2 = output2s[targets > 0.5,:].cpu().detach().numpy().copy()
                _, geniune_scores, imposter_scores = scores(embeds1, embeds2, distance_type)
                print('G', geniune_scores)
                print('IMP', imposter_scores)
                fig, plt = plot_scores(geniune_scores, imposter_scores, 'Scores', bin_width=0.05)
                fig.savefig(os.path.join(args.odir, 'tr-scores.jpg'))
                plt.close(fig)
            
            n_pids_done = min( (i_batch+1) * args.batch_size , len(train_set) )
            print('Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch+1, 
                                                                     n_pids_done,
                                                                     len(train_set),
                                                                     100. * n_pids_done / len(train_set), 
                                                                     loss.item()  ))
            
        #end batch
        logbk['train_losses'].append( np.sum(train_loss_per_batch) / (1.0 * len(train_loss_per_batch)) )

        if (epoch + 1) % args.epochs_between_saves == 0:
            torch.save(net.state_dict(), os.path.join(args.odir, 'net-ep-{}.chkpt'.format(epoch+1)) )

        # Compute validation loss & CMC:
        all_val_outputs1, all_val_outputs2 = None, None
        val_loss = 0
        net.eval()  # IMPT #
        for i_batch, data_batch in enumerate(validation_generator):
            print('{} Validation Batch [{}/ {}]'.format("{:%m-%d-%H-%M-%S}".format(datetime.now()), 
                                                        i_batch + 1, 
                                                        len(validation_generator)))
            S_pos1s, S_pos2s, S_neg1s, S_neg2s, targets_pos, targets_neg, interx_amts_pos_pair, interx_amts_neg_pair, masks_pos_pair, masks_neg_pair, pos_pids, neg_pids = data_batch
            input1s, input2s, targets = torch.cat((S_pos1s, S_neg1s)), torch.cat((S_pos2s, S_neg2s)), torch.cat((targets_pos, targets_neg))
            input1s, input2s, targets = input1s.to(device), input2s.to(device), targets.to(device) # yes! must re-assign
            output1s, output2s = net(input1s, input2s)                       
            loss = contrastive_loss(output1s, output2s, targets.float())
            val_loss += loss.item()
            # save embeddings for cmc computation later:
            embeds1 = output1s[targets > 0.5,:].cpu().detach().numpy().copy()
            embeds2 = output2s[targets > 0.5,:].cpu().detach().numpy().copy()
            if all_val_outputs1 is None:
                all_val_outputs1 = embeds1.copy()
                all_val_outputs2 = embeds2.copy()
            else:
                all_val_outputs1 = np.concatenate((all_val_outputs1, embeds1))
                all_val_outputs2 = np.concatenate((all_val_outputs2, embeds2))
            #print('all_val_outputs1 & 2 shape', all_val_outputs1.shape, all_val_outputs2.shape)
        logbk['validation_losses'].append( val_loss / (1.0 * len(validation_generator)) )
        print('Epoch: {}\tVal Loss {}'.format(epoch+1, logbk['validation_losses'][-1]))
        # Compute validation CMC:
        assert(~np.array_equal(all_val_outputs1 , all_val_outputs2))
        score_mat, _, _ = scores(all_val_outputs1, all_val_outputs2, distance_type)
        distmat = 1 - score_mat
        cmc_values = cmc_count(distmat=distmat, n_selected_labels=None, n_repeat=1)
        try:
            print('CMC ranks 1 5 10 20:', cmc_values[[0, 4, 9, 19]])
        except Exception as e:
            print(e)
        logbk['cmc'].append(cmc_values.astype(np.float16))  # save

        pickle.dump(logbk, open(os.path.join(args.odir, 'trn-logbk-{}.pkl'.format(script_start_time)), 'wb'), protocol=2)

        # Plot losses:
        fig = plt.figure()
        ax = fig.gca()
        plt.title('Train & Validation Losses')
        plt.plot(range(1, len(logbk['train_losses']) + 1), logbk['train_losses'], 'b', lw=1, label='train')
        plt.plot(range(1, len(logbk['validation_losses']) + 1), logbk['validation_losses'], 'r', lw=1, label='validation')
        plt.ylabel('Loss'); plt.xlabel('Epoch')
        ax.legend()
        #plt.show()
        fig.savefig(os.path.join(args.odir, 'loss.jpg'))
        plt.close(fig)

        # Plot CMC:
        fig = plt.figure()
        ax = fig.gca()
        plt.title('Validation CMC')
        plt.plot(range(1, len(cmc_values) + 1), cmc_values, 'b', lw=1)
        plt.ylabel('Match rate'); plt.xlabel('Rank')
        #plt.show()
        fig.savefig(os.path.join(args.odir, 'val-cmc-{}.jpg'.format(script_start_time)))
        plt.close(fig)  
        

