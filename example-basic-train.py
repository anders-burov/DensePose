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


def two_margin_cosine_embedding_loss(input1, input2, target, pos_margin=0.0, neg_margin=0.0):
    # Computes cosine loss with soft margins for positive and negative samples.
    # 
    # Example of soft margin:
    #     pos_margin = 0.1
    #     Loss contributed by a positive pair is 0 if cosine distance is greater than 0.9 (0.9 == 1 - 0.1).
    #
    # Motivation:
    #     cosine distance has range [-1, +1].
    #     similar vectors should be closer to +1.    (note: target 1)
    #     orthogonal vectors should give 0.
    #     dissimilar vectors should be closer to -1. (note: target 0)
    #     But, many a times, dissimilar/similar vectors aren't perfectly alike.
    #     For example, the same person's attire appearing slightly differently
    #     under different pose and/or illumination conditions - in such a 
    #     case, vectors representing him, v1 & v2, should be close to +1 but
    #     it may not make sense to "force" dist(v1, v2) to be exactly +1.
    #     Dissimilar vectors, v3 & v4, by the same logic, also should not be
    #     forced to be exactly -1. There could a be another dissimilar vector, v5,
    #     that is closer to, say, v3 conceptually. It is better to have dist(v5,v3) < dist(v5,v4)
    #     than "force" dist(v5,v3) == dist(v5,v4) == -1.
    #
    # Arguments:
    #     input1 & input2 are shaped (n_samples, dimensions)
    #     target is shaped (n_samples,). Either 1 or 0. 1 is positive/similar pair. 0 is negative/dissimilar pair.
    D = torch.nn.functional.cosine_similarity(input1, input2)
    device = D.device
    zeroes = torch.zeros(target.shape[0]).to(device)
    D_pos = 1.0 - pos_margin - D
    D_neg = D - neg_margin
    Z_D_pos = torch.stack([zeroes, D_pos], dim=1)
    Z_D_neg = torch.stack([zeroes, D_neg], dim=1)
    loss = (1.0*target) * torch.max(Z_D_pos, dim=1)[0]  +  (1.0 - target) * torch.max(Z_D_neg, dim=1)[0]
    loss = torch.sum(loss) / (1.0 * target.shape[0])
    return loss

def test_two_margin_cosine_embedding_loss():
    for repeat in range(100):
        i1 = torch.randn(10, 2)  * torch.randint(2, 100, (10,2))
        i2 = torch.randn(10, 2) * torch.randint(2, 100, (10,2))
        y = torch.randint(0, 2, (10,))
        l2 = two_margin_cosine_embedding_loss(i1, i2, y)
        loss1 = torch.nn.CosineEmbeddingLoss()
        y[y < 0.5] = -1
        l1 = loss1(i1, i2, y)
        d = l2 - l1
        assert(np.abs(d.numpy()) < 1e-6)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """
    def __init__(self, option='cosine', eucl_margin=1.0, margin=0.0, pos_margin=0.0, neg_margin=0.0):
        super(ContrastiveLoss, self).__init__()
        if option == 'euclidean':
            self.eucl_margin = eucl_margin
        elif option == 'cosine':
            self.loss = torch.nn.CosineEmbeddingLoss(margin=margin) # https://pytorch.org/docs/stable/nn.html#torch.nn.CosineEmbeddingLoss
        elif option == 'two margin cosine':
            test_two_margin_cosine_embedding_loss() # quick test
            self.pos_margin = pos_margin
            self.neg_margin = neg_margin
        else:
            raise Exception('No such option.')
        self.option = option

    def check_type_forward(self, in_types):
        assert len(in_types) == 3
        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))
        if self.option == 'euclidean':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1)
            dist = torch.sqrt(dist_sq)
            mdist = self.eucl_margin - dist
            dist = torch.clamp(mdist, min=0.0)
            loss = (1.0*y) * dist_sq + (1.0 - y) * torch.pow(dist, 2)
            loss = torch.sum(loss) / (2.0 * x0.size()[0])        # is really sum(loss) / 2 / n_samps
            return loss
        elif self.option == 'cosine':
            y[y < 0.2] = -1 # replace 0s with -1
            return self.loss(x0, x1, y)
        elif self.option == 'two margin cosine':
            return two_margin_cosine_embedding_loss(x0, x1, y, pos_margin=self.pos_margin, neg_margin=self.neg_margin)
        else:
            raise Exception('No such option.')
        

def scores(embeds1, embeds2, metric):
    # embeds1 and embeds2 are numpy arrays of shape (n_samples, n_features)
    # where embeds1[i,:] and embeds2[i,:] represent embeddings from
    # the SAME person.
    if metric == 'cosine':
        D = pairwise_distances(embeds1, embeds2, metric='cosine')
        D = (D - 1.0) * -1.0  # Compute similarity from distances.
        non_diag_coords = np.where(~np.eye(D.shape[0],dtype=bool))
        imposter_scores = D[non_diag_coords]
        genuine_scores = np.diagonal(D)
        return D, genuine_scores, imposter_scores  # Returns a distance matrix ("scipy convention"), genuine scores, imposter scores.
    else:
        raise NotImplementedError


def plot_scores(genuines, imposters, title, bin_width):
    fig = plt.figure()
    ax = fig.gca()
    plt.title(title)
    max_score = max(max(genuines), max(imposters))
    min_score = min(min(genuines), min(imposters))
    bins = np.arange(min_score, max_score + bin_width, bin_width)
    ax.hist(genuines, bins=bins, color='g', alpha = 0.8)
    ax.hist(imposters, bins=bins, color='r', alpha = 0.4)
    return fig, plt


# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     contrastive_loss = ContrastiveLoss(margin=1.0)
#     # for x in enumerate(train_loader):
#     #     print('len', len(x), x[0], type(x[1]))
#     #     for i in [0,1,2]:
#     #         print(i, type(x[1][i]))
#     #         print(x[1][i].shape)
#     for batch_idx, (data, input2, target) in enumerate(train_loader):     # should be bidx, (x0,x1,y)
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output1, output2 = model(data, data)                       #HACK just to get it to compile.
#         # loss = F.nll_loss(output, target)
#         loss = contrastive_loss(output1, output2, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
#             pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

# def main1():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=3, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                         help='learning rate (default: 0.01)')
#     parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                         help='SGD momentum (default: 0.5)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()

#     torch.manual_seed(args.seed)

#     device = torch.device("cuda" if use_cuda else "cpu")

#     train_dataset = datasets.MNIST('../data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ]))

#     train_iter = create_iterator(
#                                 train_dataset.data.numpy(),
#                                 train_dataset.targets.numpy(),
#                                 args.batch_size)

#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#     train_loader = torch.utils.data.DataLoader(train_iter,
#         # datasets.MNIST('../data', train=True, download=True,
#         #                transform=transforms.Compose([
#         #                    transforms.ToTensor(),
#         #                    transforms.Normalize((0.1307,), (0.3081,))
#         #                ])),
#         batch_size=args.batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=args.test_batch_size, shuffle=True, **kwargs)


#     model = Net().to(device)
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(args, model, device, test_loader)








from person_reid_nets import Siamese_Net
import resnet_custom
import torchvision.models as models
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
        
