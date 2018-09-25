from __future__ import print_function
import os
import glob
import numpy as np
import cv2
import random
import torch
from IUV_stack_utils import inds_value_of_most_prominent_person, \
                            IUV_with_only_this_inds_value, \
                            create_IUVstack, \
                            combine_IUV_stacks, \
                            get_intersection, \
                            normalize_to_reals_0to1, \
                            apply_mask_to_IUVstack

class MSMT17_V1_Load_Data_Utils():
    # Helper functions for loading data. Specific to the folder heirachy of MSMT17_V1 dataset.
    #
    # Meaning of chip name:
    #     0003_019_05_0303morning_0040_0.jpg
    #     0003 is person id within the "train" or "test" folder.
    #     019 is 20th photo (0-indexing) of this person in this "train" or "test" folder.
    #     05 is 5th cam.
    #     0303 is the date. There are 4 dates - 0303 0302 0113 0114
    #     morning, noon, afternoon
    #     _0040_0   don't know.
    #
    # Notes:
    # - Sometimes, IUV and INDS files for a particular chip won't exist. Densepose did not emit results when processing this chip.
    def __init__(self, images_train_dir, images_test_dir, denseposeoutput_train_dir, denseposeoutput_test_dir):
        self.images_train_dir = images_train_dir
        self.images_test_dir = images_test_dir
        self.denseposeoutput_train_dir = denseposeoutput_train_dir
        self.denseposeoutput_test_dir = denseposeoutput_test_dir
        self.denseposeoutput_format = '.png'
        self.train_persons_cnt = 1041
        self.test_persons_cnt = 3060
        
    def get(self, trainortest, pid, chipname):
        # example: 
        # im, IUV, INDS = get(trainortest='train', 
        #                              pid='0001', 
        #                              chipname='0001_023_05_0303morning_0032_0_ex.jpg')
        # if IUV is None or INDS is None:
        #       # skip processing.
        imdir, DPdir = self._get_image_and_DP_dirs(trainortest=trainortest)
        # print os.path.join(imdir, pid, chipname)
        im  = cv2.imread(os.path.join(imdir, pid, chipname))
        dp_IUV_name = chipname.replace('.jpg', '_IUV' + self.denseposeoutput_format)
        dp_INDS_name = chipname.replace('.jpg', '_INDS' + self.denseposeoutput_format)
        IUV = cv2.imread(os.path.join(DPdir, pid, dp_IUV_name))
        INDS = cv2.imread(os.path.join(DPdir, pid, dp_INDS_name),  0)
        assert(not im is None)
        if IUV is None or INDS is None:
            print(os.path.join(imdir, pid, chipname))
            print('IUV or INDS is None.')
        return im, IUV, INDS
    
    def example1(self):
        # You can copy and paste this to run.
        dataload = MSMT17_V1_Load_Data_Utils(images_train_dir='/data/MSMT17_V1/train', 
                                              images_test_dir='/data/MSMT17_V1/test', 
                                              denseposeoutput_train_dir='/data/IUV-densepose/MSMT17_V1/train', 
                                              denseposeoutput_test_dir='/data/IUV-densepose/MSMT17_V1/test')

        im, IUV, INDS = dataload.get('test', '0001', '0001_022_05_0303morning_0032_0.jpg')
        im, IUV, INDS = dataload.get('train', '0002', '0002_003_01_0303morning_0038_5_ex.jpg')
    
    def _get_image_and_DP_dirs(self,trainortest):
        if trainortest == 'train':
            imdir = self.images_train_dir
            DPdir = self.denseposeoutput_train_dir
        elif trainortest == 'test':
            imdir = self.images_test_dir
            DPdir = self.denseposeoutput_test_dir
        else:
            raise Exception('Unexpected arg')
        return imdir, DPdir
    
    def chips_of(self, trainortest, pid):
        # Returns chip paths of person id pid.
        imdir, _ = self._get_image_and_DP_dirs(trainortest=trainortest)
        chips = glob.glob(os.path.join(imdir, pid, '*.jpg'))
        return chips

    def num_chips(self, trainortest, pid):
        # Returns number of chips
        return len(self.chips_of(trainortest, pid))
    
    def random_chips(self, trainortest, pid, n):
        # Returns paths to n chips of this person randomly chosen.
        # All chips unique.
        # Example:
        # print(dataload.random_chips('train', '0001', 5))
        imdir, _ = self._get_image_and_DP_dirs(trainortest=trainortest)
        chips = glob.glob(os.path.join(imdir, pid, '*.jpg'))
        return np.random.choice(chips, n, replace=False)




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


def preprocess_IUV_stack(IUV_stack):
    # IUV_stack: Format is format of output of create_IUVstack(image_file, IUV_png_file).
    #            This code can be MODIFIED to also take in a HSV or HS (no V) type of IUVstack.
    # device: torch device. E.g. 'cuda' or 'cpu'.
    S = IUV_stack[:, 16:256-16, 16:256-16, :].copy()    # Crop to 24x224x224x3
    S = normalize_to_reals_0to1(S)                      # Normalize
    S = torch.Tensor(S)                                 # Convert to torch tensors
    #S = S.to(device)
    return S.view(24*3, 224, 224)





class Dataset_Test(torch.utils.data.Dataset):
    def __init__(self, precomputed_test_path, setid, glabels, plabels):
        """
        Args:
            precomputed_test_path (string): e.g. '/data/IUV-densepose/MSMT17_V1/precomputed'
            glabels: pids of gallery people e.g. [12, 44, 55, 37, 897, 1034, ... ]
            plabels: pids of probe people e.g. [12, 44, 55, 37]
        """
        super(Dataset_Test, self).__init__()
        assert(all([pid == glabels[i] for i, pid in enumerate(plabels)]))
        self.glabels = list(glabels) # deep-copy or convert to list
        self.plabels = list(plabels) # deep-copy or convert to list
        glabels, plabels = None, None  # safety
        self.gpicks = []
        self.ppicks = []
        for i, gpid in enumerate(self.glabels):
            if i < len(self.plabels): 
                ppid = self.plabels[i]
                assert(gpid == ppid)
                picks = np.random.permutation([1, 2])
                self.gpicks.append(picks[0])
                self.ppicks.append(picks[1])
            else:
                assert(i >= len(self.plabels))
                self.gpicks.append(random.randint(1,2))
        assert(len(self.gpicks) == len(self.glabels))
        assert(len(self.ppicks) == len(self.plabels))
        self.shape_of_similarity_matrix = (len(self.glabels), len(self.plabels)) # This is the convention by cysu's cmc.
        # shape_of_similarity_matrix convention:
        # going down rows is going thru gallery ppl.
        # going along columns is going thru probe ppl.
        # ----
        self.precomputed_test_path = precomputed_test_path
        self.setid = setid

    def __len__(self):
        return self.shape_of_similarity_matrix[0] * self.shape_of_similarity_matrix[1]

    def read_IUV_stack(self, setid, pid, pick, astypefloat64=True):
        # e.g. setid = 1, pid = 432, pick = 1 or 0
        IUVs = np.load(os.path.join(self.precomputed_test_path, str(setid), str(pid)+'.npz'  ))
        S = IUVs['S'+str(pick)].copy()
        return S
        # return S.astype(np.float) if astypefloat64 else S

    def __getitem__(self, index):
        row_idx, col_idx = np.unravel_index(indices=index, 
                                            dims=self.shape_of_similarity_matrix, order='F') # F means assume column-major.
        # Convention:
        # go down rows is go thru gallery ppl.
        # go along columns is go thru probe ppl.
        # advancing index is go down rows. If at last row, go to first row of next column.
        g_pid = self.glabels[row_idx]
        g_pick = self.gpicks[row_idx]
        p_pid = self.plabels[col_idx]
        p_pick = self.ppicks[col_idx]
        S_gal = self.read_IUV_stack(self.setid, g_pid, g_pick, False)
        S_probe = self.read_IUV_stack(self.setid, p_pid, p_pick, False)
        # S_gal = S_gal.astype(np.float)
        # S_probe = S_probe.astype(np.float)
        mask = get_intersection(S_gal, S_probe)
        intersection_amt = np.sum(mask.astype(np.int))
        # most_info_in_an_input_so_far = max(np.sum(mask), most_info_in_an_input_so_far) # update
        # print('% IUV filled: ', 1.0 * np.sum(mask) / most_info_in_an_input_so_far)
        Sp = apply_mask_to_IUVstack(S_probe.copy(), mask)
        Sg = apply_mask_to_IUVstack(S_gal.copy(), mask)
        Sp = preprocess_IUV_stack(Sp)#, device='cpu')
        Sg = preprocess_IUV_stack(Sg)#, device='cpu')
        target = 1 if g_pid==p_pid else 0
        # target = torch.Tensor([target])
        # return [100], [200], target, 400, 500, 600, 700,  row_idx, col_idx, intersection_amt, index
        return Sg, Sp, target, g_pid, p_pid, g_pick, p_pick, row_idx, col_idx, intersection_amt, index
        #Sg is IUV stack of gallery. Sp is IUV stack of probe.




class Dataset_msmt17(torch.utils.data.Dataset):
    def __init__(self, dataload=None, trainortest=None, pids_allowed=None, mask_inputs=False, combine_mode='average v2'):
        """
        Args:
            dataload: A MSMT17_V1_Load_Data_Utils object.
            trainortest: 'train' or 'test'
            pids_allowed: e.g. [0,1,2,3,..., 1040]
            mask_inputs: True/False. If True, input pairs are masked
                         such that only the common regions remain.
                         Note that even though this "sanitizes the input",
                         it makes the computational complexity of resolving 
                         P probes and G galleries O(PG). Without masking,
                         it is O(P+G).
            combine_mode: the arg combine_mode in combined_IUVstack_from_multiple_chips
        """
        super(Dataset_msmt17, self).__init__()
        if dataload is None:
            dataload = msmt17_v1_utils.MSMT17_V1_Load_Data_Utils(images_train_dir='/data/MSMT17_V1/train', 
                                                            images_test_dir='/data/MSMT17_V1/test', 
                                                            denseposeoutput_train_dir='/data/IUV-densepose/MSMT17_V1/train', 
                                                            denseposeoutput_test_dir='/data/IUV-densepose/MSMT17_V1/test')
        self.dataload = dataload
        self.trainortest = trainortest
        assert(not pids_allowed is None)
        self.pids_allowed = pids_allowed
        self.mask_inputs = mask_inputs
        self.combine_mode = combine_mode

    def __len__(self):
        return len(self.pids_allowed)

    def positive_pair(self, pid):
        pidstr = str(pid).zfill(4)
        n_chips_avail = self.dataload.num_chips(self.trainortest, pidstr)
        chips = self.dataload.random_chips(self.trainortest, pidstr, n_chips_avail)
        split1 = chips[ : int(len(chips)/2)]
        split2 = chips[int(len(chips)/2) : ]
        S1 = combined_IUVstack_from_multiple_chips(self.dataload, pid=pidstr, chip_paths=split1, trainortest=self.trainortest, combine_mode=self.combine_mode)
        S2 = combined_IUVstack_from_multiple_chips(self.dataload, pid=pidstr, chip_paths=split2, trainortest=self.trainortest, combine_mode=self.combine_mode)
        mask = get_intersection(S1, S2)    # intersect
        return S1, S2, mask

    def one_negative_sample(self, pos_pid, S_pos):
        assert(pos_pid in self.pids_allowed)
        persons = list(self.pids_allowed)
        persons.remove(pos_pid)
        neg_pid = random.choice(persons)
        assert(neg_pid != pos_pid) # to prevent edge cases.
        pidstr = str(neg_pid).zfill(4)
        n_chips_avail = self.dataload.num_chips(self.trainortest, pidstr)
        chips = self.dataload.random_chips(self.trainortest, pidstr, n_chips_avail)
        split = chips[ : int(len(chips)/2)]
        S_neg = combined_IUVstack_from_multiple_chips(self.dataload, pid=pidstr, chip_paths=split, trainortest=self.trainortest, combine_mode=self.combine_mode)
        mask = get_intersection(S_neg, S_pos)    # intersect
        return S_neg, S_pos, mask, neg_pid

    def __getitem__(self, index):
        #
        pos_pid = self.pids_allowed[index]
        S1, S2, mask_pos_pair = self.positive_pair(pid=pos_pid)
        interx_amt_pos_pair = np.sum(mask_pos_pair)
        #
        S_neg, S_neg_2, mask_neg_pair, neg_pid = self.one_negative_sample(pos_pid=pos_pid, S_pos=S1)
        interx_amt_neg_pair = np.sum(mask_neg_pair)
        #
        if self.mask_inputs:
            S1 = apply_mask_to_IUVstack(S1, mask_pos_pair)
            S2 = apply_mask_to_IUVstack(S2, mask_pos_pair)
            S_neg = apply_mask_to_IUVstack(S_neg, mask_neg_pair)
            S_neg_2 = apply_mask_to_IUVstack(S_neg_2, mask_neg_pair)
        #
        S_pos1 = preprocess_IUV_stack(S1)
        S_pos2 = preprocess_IUV_stack(S2)
        S_neg1 = preprocess_IUV_stack(S_neg)
        S_neg2 = preprocess_IUV_stack(S_neg_2)
        target_pos = 1 # don't use np.array([1])
        target_neg = 0
        #
        return S_pos1, S_pos2, S_neg1, S_neg2, target_pos, target_neg, interx_amt_pos_pair, interx_amt_neg_pair, mask_pos_pair, mask_neg_pair, pos_pid, neg_pid
