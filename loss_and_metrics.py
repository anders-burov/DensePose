from __future__ import print_function
import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import torch
import torch.nn as nn

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


def adaptive_margins(confidence, type, **kwargs):
    # Args:
    #  confidence: numpy array or torch tensor. shape (n_samps,). Range: [0,1] reals.
    #  type: see code below.
    #  kwargs: see code below.
    # Returns:
    #  positive_margins: numpy array or torch tensor. shape (n_samps,).
    #  negative_margins: same as positive_margins.
    if type == 'cosine':
        posmax_rad = kwargs['max_positive_margin_angle_rad'] # Range should be 0 to 90 degrees == 0 to pi/2 radians.
        posmin_rad = kwargs['min_positive_margin_angle_rad']
        negmax_rad = kwargs['max_negative_margin_angle_rad']
        negmin_rad = kwargs['min_negative_margin_angle_rad']
        g1 = 1.0 * (posmin_rad - posmax_rad)
        g2 = 1.0 * (negmin_rad - negmax_rad)
        positive_margin_rad = g1 * confidence + posmax_rad
        negative_margin_rad = g2 * confidence + negmax_rad
        positive_margin = 1 - np.cos(positive_margin_rad)
        negative_margin = np.cos(np.deg2rad(90) - negative_margin_rad)
        return positive_margin, negative_margin
    else:
        raise NotImplementedError


def test_adaptive_margins():
    confidences = np.array([0.0, 0.2, 0.5, 0.75, 1.0])
    kwargs = {'max_positive_margin_angle_rad':np.deg2rad(90),
              'min_positive_margin_angle_rad':np.deg2rad(5),
              'max_negative_margin_angle_rad':np.deg2rad(90),
              'min_negative_margin_angle_rad':np.deg2rad(20)}
    positive_margins, negative_margins = adaptive_margins(confidences, 'cosine', **kwargs)
    print('test_adaptive_margins() ...')
    print(positive_margins, negative_margins)
    assert(all(x > xnext for x, xnext in zip(positive_margins, positive_margins[1:])))
    assert(all(x > xnext for x, xnext in zip(negative_margins, negative_margins[1:])))


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


class Adaptive_Double_Margin_Contrastive_Loss(torch.nn.Module):
    
    def __init__(self, option, **kwargs):
        # To know what to input in kwargs, see the code of the loss function you're choosing.
        super(ContrastiveLoss, self).__init__()
        if option == 'euclidean':
            raise NotImplementedError
        elif option == 'adaptive double margin cosine':
            test_adaptive_margins()
            self.kwargs = kwargs
        else:
            raise Exception('No such option.')
        self.option = option

    def check_type_forward(self, in_types):
        assert len(in_types) == 4
        x0_type, x1_type, y_type confi_type = in_types
        assert x0_type.size() == x1_type.shape
        assert confi_type.size() == y_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y, confidence):
        self.check_type_forward((x0, x1, y, confidence))
        if self.option == 'adaptive double margin cosine':
            pos_margin, neg_margin = adaptive_margins(confidence, 'cosine', **self.kwargs)
            return two_margin_cosine_embedding_loss(x0, x1, y, pos_margin=pos_margin, neg_margin=neg_margin)
        else:
            raise Exception('No such option.')




def scores(embeds1, embeds2, metric):
    # embeds1 and embeds2 are numpy arrays of shape (n_samples, n_features)
    # where embeds1[i,:] and embeds2[i,:] represent embeddings from
    # the SAME person.
    if metric == 'cosine':
        #assert(np.array_equal(embeds1, embeds2))
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