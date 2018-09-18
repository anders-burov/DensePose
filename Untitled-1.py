import msmt17_v1_utils
import torch
import numpy as np
import math

def soft_cosine_embedding_loss(input1, input2, target, pos_margin=0.0, neg_margin=0.0):
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
    zeroes = torch.zeros(target.shape[0])
    D_pos = 1.0 - pos_margin - D
    D_neg = D - neg_margin
    Z_D_pos = torch.stack([zeroes, D_pos], dim=1)
    Z_D_neg = torch.stack([zeroes, D_neg], dim=1)
    loss = (1.0*y) * torch.max(Z_D_pos, dim=1)[0]  +  (1.0 - y) * torch.max(Z_D_neg, dim=1)[0]
    loss = torch.sum(loss) / (1.0 * target.shape[0])
    return loss


i1 = torch.randn(10, 2)
i2 = torch.randn(10, 2)
# print(torch.max(i1, dim=1)[0])
# exit()
y = torch.randint(0, 2, (10,))
print y

l2 = soft_cosine_embedding_loss(i1,i2,y,pos_margin=0.0)

loss1 = torch.nn.CosineEmbeddingLoss(margin=0)
y[y<0.5] = -1
l1 = loss1(i1,i2,y)
# print(l1)
# print y
print 'l1'
print l1
d=l2-l1
print np.abs(d.numpy())
assert(np.abs(d.numpy()) < 1e-7)
exit()



output = torch.nn.functional.cosine_similarity(input1, input2)
output[output<0] = 999
print(output, len(output))
exit()

T1 = torch.randn(24,224,224,3) 
T2 = torch.randn(24,224,224,3)

T1.to('cpu')
T2.to('cpu')

L=[T1,T2]

S = torch.stack(L)
print S.shape
x=S.view(2,24*3,224,224)

y = torch.clamp(x, -5, min=0.0)

print x.shape


# import torchvision

# cc = torchvision.transforms.CenterCrop(size=224)
# Tc = cc((T2))
# print Tc.shape


# dataload = msmt17_v1_utils.MSMT17_V1_Load_Data_Utils(images_train_dir='/data/MSMT17_V1/train', 
#                                                             images_test_dir='/data/MSMT17_V1/test', 
#                                                             denseposeoutput_train_dir='/data/IUV-densepose/MSMT17_V1/train', 
#                                                             denseposeoutput_test_dir='/data/IUV-densepose/MSMT17_V1/test')

# pidstr=str(0).zfill(4)
# chips_avail = dataload.num_chips('train', pidstr)
# chip_paths = dataload.chips_of('train', pidstr)
# print chip_paths, len(chip_paths)


