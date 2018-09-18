import cv2
import matplotlib.pyplot as plt
import numpy as np
from distutils.version import StrictVersion
assert(StrictVersion(np.__version__) >= StrictVersion('1.14.5')) # to be safe.


def get_XYUV_pts_of_a_bodypart(IUV, idx):
    # idx is SMPL index. There are 24 body parts.
    xs, ys =np.where(IUV[:,:,0] == idx)
    us = IUV[xs, ys, 1]
    vs = IUV[xs, ys, 2]
    return xs, ys, us, vs


def intersecting_UVs(us1, vs1, us2, vs2):
    C1 = [(u,v) for u,v in zip(us1,vs1)]
    C2 = [(u,v) for u,v in zip(us2,vs2)]
    common_UVs = set(C1) & set(C2)
    us, vs = np.hsplit(np.array(list(common_UVs)), 2)  # 2 arrays: u-coords, v-coords
    return np.ndarray.flatten(us), np.ndarray.flatten(vs)


def get_color_at(im, IUV, bodypart, u, v):
    # image is hwc e.g. shaped (1920,1080,3)
    # IUV is hwc e.g. shaped (1920,1080,3). Last axis holds I,U,V
    # bodypart is index out of 24 body parts. scalar
    # u and v are 0~255.
    # Returns colors. May be 1 color like: [[150, 65,43]]. May be multiple colors like:
    # [[150, 65,43], [150,63,43], [151,64,44]] since multiple pixels may be mapped to the
    # same (i, u, v) coordinate.
    colors_array = im[(IUV[:,:,0] == bodypart) & (IUV[:,:,1] == u) & (IUV[:,:,2] == v)]
    return colors_array


def get_intersection(A_, B_):
    # A_, B_ are IUV stacks where
    # cells with values 0-255 has some info and cells
    # with values -1 have no info.
    # Intersection of A & B is cells which
    # possess info.
    #
    # Returns a mask whose cells either contain 1 or 0.
    # 1 denotes intersection. 0 for outside intersection.
    A = A_.copy()
    B = B_.copy()
    A[A>=0] = 1 # create mask. Mask symbol: 1 info present. 0 absent.
    A[A<0] = 0
    B[B>=0] = 1 # create mask.
    B[B<0] = 0
    return A * B # 1 intersecting cells. 0 outside of intersection.


def apply_mask_to_IUVstack(S_, M):
    # S_ is IUV stack
    # M is mask of 0s & 1s.
    # Does not modify S_.
    S = S_.copy()
    S[M == 0] = -1
    return S


def test_intersection_IUVstacks():
    A = np.array([-1, 4, -1, 0, 0])
    B = np.array([5, 6, -1, 0, -1])
    M = get_intersection(A, B)
    am = apply_mask_to_IUVstack(A, M)
    bm = apply_mask_to_IUVstack(B, M)
    assert((A == np.array([-1, 4, -1, 0, 0])).all() )
    assert((M == np.array([0, 1, 0, 1, 0])).all() )
    assert((am == np.array([-1, 4, -1, 0, -1])).all() )
    assert((bm == np.array([-1, 6, -1, 0, -1])).all() )

test_intersection_IUVstacks()


def create_IUVstack(im, IUV):
    # im: shaped (h,w,3). Up to you to choose color convention- RGB, HSV, etc. .
    # IUV: shaped (h,w,3). 3 is for (I, U, V) values.
    # Returns a (24, 256, 256, 3) shaped array.
    # There are 24 different SMPL parts.
    # 256 x 256 is h x w of each SMPL part flattened.
    # 3 is the color chans follownig the convention of argument 'im'.
    A = np.append(im, IUV, axis=-1)
    A = A.reshape(A.shape[0] * A.shape[1], 6)
    #print(A.shape)
    E = A[(A[:, 3] > 0)]  # condition that index I of (I,U,V) > 0. 0 means no body part at that picture location.
    #print(E.shape)
    stack = np.ones((24, 256, 256, 3)) * -1     # this is the IUV stack. unfilled pixels eventually assigned -1.
    for body_part_idx in range(1, 25):
        Q = E[(E[:, 3] == body_part_idx)]
        #print(body_part_idx, Q.shape)
        if Q.shape[0] > 0:
            stack[body_part_idx - 1, Q[:, 4], Q[:, 5], :] = Q[:, 0:3]
    return stack


def plot_IUVstack(stack_, background_rgb=128, figsize=(6,6)):
    # hard coded this to show only RGB typed stack properly.
    # HSV will show in a messed up manner.
    # background_rgb: 0 black, 128 grey, 255 white.
    stack = stack_.copy()
    stack[stack < 0] = background_rgb
    for i in range(24):
        fig = plt.figure(figsize=figsize)
        plt.title('smpl index %d'%(i))
        plt.imshow(stack[i,:,:,::-1] / 255.0)
        plt.show()


def L1_norm_normalized_by_avail_data(A, B):
    # Computes distance between A and B IUV stacks.
    # A & B are IUV stacks.
    # "L1 norm" is | x - y | kinda distance.
    # "normalized by avail data" means divided by data in intersection of A and B stacks.
    M = get_intersection(A, B)
    Am = apply_mask_to_IUVstack(A, M)
    Bm = apply_mask_to_IUVstack(B, M)
    L1_norm = np.sum(np.abs(Am - Bm))
    num_avail_data = np.sum(M)
    return L1_norm / num_avail_data


def rgb_distance(A, B):
    # Computes distance between A and B IUV stacks.
    # A & B are IUV stacks.
    # distances are normalized by amount of data available in the intersection of A and B.
    # distance between 2 RGB pixels is defined here as L2 norm.
    # return:
    #    1. Total RGB distance, scalar. Normalized over amt of avail data in all body parts.
    #    2. List of RGB distances of each body part (24 body parts). Normalized over amount of avail data in that particular body part.
    M = get_intersection(A, B)
    # print('M', M.shape)
    Am = apply_mask_to_IUVstack(A, M)
    Bm = apply_mask_to_IUVstack(B, M)
    X = np.square(Am - Bm)
    distances_bodypart_wise = []
    for i in range(X.shape[0]):
        sum_along_rgb_axis = np.sum(X[i,:,:,:], axis=-1)
        sqrooted = np.sqrt(sum_along_rgb_axis)
        # print('sqrooted.shape', sqrooted.shape)
        total = np.sum(sqrooted)
        avail_data = np.sum(M[i,:,:,:]) / 3  # divided by 3 cuz rgb.
        # print(avail_data)
        if avail_data > 0:
            distances_bodypart_wise.append(total / avail_data)
        else:
            distances_bodypart_wise.append(0)
    # print distances_bodypart_wise
    np.array(distances_bodypart_wise)
    total_dist = np.sum(np.array(distances_bodypart_wise)) / len(distances_bodypart_wise)
    return total_dist, distances_bodypart_wise


def inds_values_and_their_pixel_counts(instance_seg_mask, exclude_foreground=True):
    # Example:
    #     instance_seg_mask = cv2.imread('/data/IUV-densepose/MSMT17_V1/train/0001/0001_021_07_0303morning_0035_1_INDS.png',  0)
    #     inds_val_list, pixel_cnt_list = inds_values_and_their_pixel_counts(instance_seg_mask)
    #     inds_val_list is like [8, 9].
    #     pixel_cnt_list is like [1661, 760]. There are 1661 pixels associated with instance 8. 760 pixels for instance 9.
    #     Notice no inds val 0. 0 is foreground and is excluded by default.
    idxs_list = np.unique(instance_seg_mask)  # output is like [0, 8, 12]
    output_inds = []
    output_counts = []
    for i in idxs_list:
        if exclude_foreground and i == 0:
            continue
        C = np.where(instance_seg_mask == i)
        a_person = instance_seg_mask[C[0], C[1]]
        #print(np.unique(a_person))
        output_inds.append(i)
        output_counts.append(len(a_person))
    # assert(np.sum(counts) == instance_seg_mask[0] * instance_seg_mask[1]) # test. msut include foreground if you wanna try this test!s
    return output_inds, output_counts


def inds_values_and_their_bboxs(instance_seg_mask, exclude_foreground=True):
    # Example:
    #     instance_seg_mask = cv2.imread('/data/IUV-densepose/MSMT17_V1/train/0001/0001_021_07_0303morning_0035_1_INDS.png',  0)
    #     inds_val_list, bbox_list = inds_values_and_their_bbox(instance_seg_mask)
    #     inds_val_list is like [8, 9].
    #     bbox_list is like [(36, 134, 25, 49), (1, 58, 26, 50)].
    #                       36 is xmin, 134 is xmax, 25 is ymin, 49 is ymax.
    #     Notice no inds val 0. 0 is foreground and is excluded by default.
    idxs_list = np.unique(instance_seg_mask)  # output is like [0, 8, 12]
    output_inds = []
    output_bboxs = []
    for i in idxs_list:
        if exclude_foreground and i == 0:
            continue
        C = np.where(instance_seg_mask == i)
        xmin = np.min(C[0])
        xmax = np.max(C[0])
        ymin = np.min(C[1])
        ymax = np.max(C[1])
        output_inds.append(i)
        output_bboxs.append( (xmin, xmax, ymin, ymax) )
    return output_inds, output_bboxs


def inds_value_of_most_prominent_person(instance_seg_mask, mode):
    # Returns a single inds value of most prominent person. Otherwise return None.
    # Example:
    #     instance_seg_mask = cv2.imread('/data/IUV-densepose/MSMT17_V1/train/0001/0001_021_07_0303morning_0035_1_INDS.png',  0)
    #     inds_val = inds_value_of_most_prominent_person(instance_seg_mask, mode='strict')
    #     other modes include 'most bbox area', 'most pixels'.
    indxs, cnts = inds_values_and_their_pixel_counts(instance_seg_mask)
    if not indxs:
        return None # no person(s) segmented
    j = np.argmax(cnts)
    inds_val_with_most_pixels = indxs[j]
    indxs, bboxs = inds_values_and_their_bboxs(instance_seg_mask)
    if not indxs:
        return None # no person(s) segmented
    def bbox_area(bbox):
        return (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
    areas = [bbox_area(b) for b in bboxs]
    k = np.argmax(areas)
    inds_val_with_most_area = indxs[k]
    if inds_val_with_most_pixels == inds_val_with_most_area   and   mode == 'strict':
        return inds_val_with_most_pixels
    elif mode == 'most bbox area':
        return inds_val_with_most_area
    elif mode == 'most pixels':
        return inds_val_with_most_pixels
    else:
        return None


def IUV_with_only_this_inds_value(IUV, instance_seg_mask, inds_value, fail_when_no_pixels_at_this_inds_val=True):
    # Filter IUV such that only body parts associated with inds value inds_value remain.
    # IUV (h,w,3) shaped. 3 chans for I,U,V
    # instance_seg_mask is (h,w) shaped. values are person instances.
    # inds_value: selected inds value.
    # Example:
    #   im  = cv2.imread('/data/MSMT17_V1/train/0002/0002_002_01_0303morning_0038_5.jpg')
    #   IUV = cv2.imread('/data/IUV-densepose/MSMT17_V1/train/0002/0002_002_01_0303morning_0038_5_IUV.png')
    #   INDS = cv2.imread('/data/IUV-densepose/MSMT17_V1/train/0002/0002_002_01_0303morning_0038_5_INDS.png',  0)
    #   ii = inds_value_of_most_prominent_person(INDS, mode='strict')
    #   new_IUV = IUV_with_only_this_inds_value(IUV, INDS, inds_value=ii)
    #   S = create_IUVstack(im, new_IUV)
    C = np.where(instance_seg_mask == inds_value)
    #print(len(C), C)
    if fail_when_no_pixels_at_this_inds_val and len(C[0]) == 0:
        raise Exception("No pixels at inds value {}.".format(inds_value))
    new_IUV = np.zeros_like(IUV) ; #print(new_IUV.shape)
    new_IUV[C[0], C[1], :] = IUV[C[0], C[1], :].copy()
    return new_IUV


def plot_im_of_this_inds_value(im, instance_seg_mask, inds_value, background_rgb=255):
    C = np.where(instance_seg_mask == inds_value)
    im = im.copy()
    im_filt = np.ones_like(im) * background_rgb
    im_filt[C[0],C[1],:] = im[C[0],C[1],:]
    fig = plt.figure()
    plt.imshow(im_filt[:,:,::-1] / 255.0)
    plt.show()


def combine_IUV_stacks(IUV_stack_list, mode='average v2', warn=True):
    # Example:
    #    s1 = create_IUVstack(im1, IUV1)
    #    s2 = create_IUVstack(im2, IUV2)
    #    s3 = create_IUVstack(im3, IUV3)
    #    s_combined = combine_IUV_stacks( IUV_stack_list = [s1,s2,s3] )
    #    Warning: Up to you to ensure that IUV1, IUV2, IUV3 used in create_IUVstack has only 1 person.
    #             There are functions that help to isolate a particualr person's instance in a IUV array.
    #             Isolate, then create the IUV_stack. Doesn't quite make sense to have multiple peple in a 
    #             single IUV stack.
    if mode == 'average':
        raise Exception('wrong!')
        s_combined = np.zeros_like(IUV_stack_list[0])  # init
        for s in IUV_stack_list:
            s_combined += s                   # WRONG for the case where   -1 + (>=0)
        return s_combined / len(IUV_stack_list)   # WRONG for the case where  ( -1 + (>=0) ) /  2
    elif mode == 'average v2' or mode is None:
        if warn and len(IUV_stack_list) > 250:
            raise Exception('You might wish to combine fewer stacks due to numerical stability issues.')
        # Count how many IUV stacks in IUV_stack_list contain info for all IUV stack coordinates:
        info_counts = np.zeros_like(IUV_stack_list[0])
        for s in IUV_stack_list:
            info_counts[s > -0.2] += 1  # increment count when encounter information.
        # Suppose info_counts[12, 251, 136, 2] == 5. It means that 5 IUV stacks contain info at coordinate (12, 251, 136, 2).
        info_mask = np.zeros_like(IUV_stack_list[0])
        info_mask[info_counts > 0] = 1
        info_counts[info_counts < 1] = 1 # so that later "won't divide by zero" error when info_counts is denominator.
        # -------
        # Sum all IUV stacks:
        NANified_IUVstacks = np.stack(IUV_stack_list, axis=0)   # (n x 24 x 256 x 256 x 3) shape. n is number of IUV stacks in IUV_stack_list.
        NANified_IUVstacks[NANified_IUVstacks < -1e-2] = np.nan
        s_combined = np.nansum(NANified_IUVstacks, axis=0)  # np.nansum effects: nan + <num> = <num>. nan + nan = 0.
        # at this stage, s_combined has 0s where no info AND where r/g/b is 0 value. AMBIGUOUS!
        # To resolve ambiguity, assign NaN to places with no info instead:
        s_combined[info_mask < 0.5] = np.nan
        s_combined = s_combined / info_counts               # normalize. note: NaN / <num> == NaN.
        s_combined[np.isnan(s_combined)] = -1               # respect convention that elements absent of information are denoted -1.
        return s_combined
    elif mode == 'overwrite':
        s_combined = np.ones_like(IUV_stack_list[0]) * -1 # init
        for s in IUV_stack_list:
            C = np.where(s > -1e-2)
            # s_combined[C[0], C[1], C[2], C[3]] = s[C[0], C[1], C[2], C[3]]
            s_combined[C] = s[C]
        return s_combined
    elif mode == 'weighted average':
        # combined :=  x * w1 + new * w2  where sum(w_i) == 1.
        raise NotImplementedError
    else:
        raise Exception('Unrecognized mode.')


def normalize_to_reals_0to1(IUV_stack):
    # Normalizes an IUV stack to range in reals [0, 1].
    # 0 now denotes no info.
    # maps range [-1, 255] to reals in [0, 1].
    return (IUV_stack + 1.0) / (255.0 - (-1.0))

def num_elements_can_hold_info():
    # Number of elements in 24 x 256 x 256 x 3 IUV stack that can contain info.
    return 3615531 # Empirically found on MSMT17_V1 dataset.