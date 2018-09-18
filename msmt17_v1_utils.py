import os
import glob
import numpy as np
import cv2


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
            print os.path.join(imdir, pid, chipname)
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