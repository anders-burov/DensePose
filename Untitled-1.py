import msmt17_v1_utils
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import sklearn
print(sklearn.__file__)

train_loss_record= [99, 34, 0.8]

fig = plt.figure()
ax = fig.gca()
plt.title('train & val loss')
np.arange(1, len(train_loss_record))
plt.plot(range(1, len(train_loss_record)+1), train_loss_record, 'b')
plt.ylabel('loss'); plt.xlabel('epochs')
plt.show()
# plt.plot(list(range(1, len(train_loss_record+1))), train_loss_record, 'b')

exit()





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


def plot_scores(genuines, imposters, title, n_bins=100):
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca()
    plt.title(title)
    ax.hist(genuines, bins=n_bins, alpha = 0.5)
    ax.hist(imposters, bins=int(n_bins*1.2), color='r', alpha = 0.3)
    return fig, plt



# geniune and imposter distances:

torch.set_printoptions(precision=10)

i1s = torch.randn(10, 2) #* torch.randint(2, 100, (10,2))
i2s = torch.randn(10, 2) #* torch.randint(2, 100, (10,2))
D = torch.nn.functional.cosine_similarity(i1s, i2s)
print('D', D)
D2 = pairwise_distances(i1s.numpy(), i2s.numpy(), metric='cosine')
D2 = (D2 - 1.0) * -1.0  # Compute similarity from distances.
non_diag_coords = np.where(~np.eye(D2.shape[0],dtype=bool))
imposter_scores = D2[non_diag_coords]
print('imopster')
print(len(imposter_scores))
print(imposter_scores)
genuine_scores = np.diagonal(D2)
diag = genuine_scores
print('D2 diag', diag)
print('D2 imposter distances:')
print(D2[0,1], torch.nn.functional.cosine_similarity(i1s[[0],:], i2s[[1],:]))
print(D2[0,2], torch.nn.functional.cosine_similarity(i1s[[0],:], i2s[[2],:]))
print(D2[0,3], torch.nn.functional.cosine_similarity(i1s[[0],:], i2s[[3],:]))



dist_mat, geniune_scores, imposter_scores = scores(i1s, i2s, 'cosine')
fig, plt = plot_scores(geniune_scores, imposter_scores, 'tile123')
fig.savefig('delme.jpg')



# plt.show();
# plt.show(block=False)
# plt.draw(); 
# import time; time.sleep(100)
# print('asdf')

exit()

# im

# T1 = torch.randn(24,224,224,3) 
# T1 = T1.view(T1.shape[0] * T1.shape[-1], 224, 224)
# print T1.shape

# exit()

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


