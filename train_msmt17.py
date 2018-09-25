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
from IUV_stack_utils import *  #TODO
import msmt17_v1_utils
from loss_and_metrics import ContrastiveLoss, scores, plot_scores


if __name__ == '__main__':
    
    # Injecting intersection suffiency check.   IUV utils > intersection check & n_elements_can_hold_info.  Basically redo if cannot find sufficient intersection.

    # argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--odir', type=str, default='tmp',
                        help='output dir. e.g. \'expt-1\'.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='gpu or cpu device. cuda cuda:0 cuda:1 cpu')
    parser.add_argument('--batch_size', type=int, default=50,
                        help="If batch size is 4, there\'ll be 4 positive pairs and 4 negative pairs per batch. Implies 8 pairs per batch.")
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num workers on dataloaders.')
    parser.add_argument('--overfit', type=bool, default=True,
                        help='')                       
    args = parser.parse_args()
    
    logbk = {}

    val_samples = 3
    epochs = 5000
    epochs_between_saves = 10
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
    
    logbk['train_pids_allowed'] = tuple(range(dataload.train_persons_cnt))
    logbk['val_pids_allowed'] = tuple(range(dataload.train_persons_cnt))

    if args.overfit:
        print('OVERFIT!' * 20)
        logbk['train_pids_allowed'] = tuple([4,8,12,13,14])
        logbk['val_pids_allowed'] = tuple([2000, 3000])
        args.batch_size = 3

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
    logbk['intersection_amt_stats'] = Welford()
    logbk['intersection_amt_max'] = 0

    #most_info_in_an_input_so_far = 0.03 * (24 * 224 * 224 * 3) # 224 is h,w accepted into net. init.
    
    for epoch in range(epochs):
        train_loss_per_batch = [] 
        val_loss_1_epoch_tmp_record = []
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

            if plot_training_metric:
                embeds1 = output1s[targets > 0.5,:].detach().numpy().copy()
                embeds2 = output2s[targets > 0.5,:].detach().numpy().copy()
                _, geniune_scores, imposter_scores = scores(embeds1, embeds2, distance_type)
                print('G', geniune_scores)
                print('IMP', imposter_scores)
                fig, plt = plot_scores(geniune_scores, imposter_scores, 'Scores', bin_width=0.05)
                fig.savefig(os.path.join(args.odir, 'tr-scores.jpg'))
                plt.close(fig)
            
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch+1, 
                                                                     max( (i_batch+1) * args.batch_size , len(train_set) ),
                                                                     len(train_set),
                                                                     100. * (i_batch+1) / args.batch_size, 
                                                                     loss.item()  ))
            
        #end batch
        logbk['train_losses'].append( np.sum(train_loss_per_batch) / (1.0 * len(train_loss_per_batch)) )

        # insert save logbk here.

        if (epoch + 1) % epochs_between_saves == 0:
            torch.save(net.state_dict(), os.path.join(args.odir, 'net-ep-{}.chkpt'.format(epoch+1)) )

        if plot_loss:
            fig = plt.figure()
            ax = fig.gca()
            plt.title('train & val loss')
            plt.plot(range(1, len(logbk['train_losses']) + 1), logbk['train_losses'], 'b')
            plt.ylabel('loss'); plt.xlabel('epochs')
            fig.savefig(os.path.join(args.odir, 'loss.jpg'))
            plt.close(fig)
        
        
    


exit()
# compute val loss.
# compute val cmc.
# plot val loss






for x in wtf:


    for epoch in range(epochs):
        
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



exit()


#------------------------------------------------------------------
for x in wtf:
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
        
