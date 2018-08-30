##############################################################################
# Perform, and save inference on person reidentification datasets.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import subprocess

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

import imp
this_py_file_dir = os.path.dirname(os.path.realpath(__file__))
prutils = imp.load_source('', this_py_file_dir + '/' + '../person_reid_utils.py')

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--reid-dataset',
        dest='reid_dataset',
        help='example is MSMT17_V1',
        default='MSMT17_V1',
        type=str
    )
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def make_output_dir_path(output_dir, im_name, dataset):
    # Returns the output dir path that is like   /whatever/MSMT17_V1/train/0000  .
    if dataset == 'MSMT17_V1':
        i = im_name.find('MSMT17_V1')
        assert(i > -1) # cehck
        suffix = os.path.dirname(im_name[i:])  # example MSMT17_V1/train/0000
        extended_output_dir = os.path.join(output_dir, suffix)
        cmd = "mkdir -p " + extended_output_dir
        subprocess.Popen(cmd.split())
        return extended_output_dir
    elif dataset is None:
        return output_dir
    else:
        raise Exception

def make_output_dirs(output_dir, im_or_folder, dataset):
    # Makes output dirs.
    if dataset == 'MSMT17_V1':
        cmd = "mkdir -p " + output_dir + '/MSMT17_V1/train'
        subprocess.Popen(cmd.split())
        cmd = "mkdir -p " + output_dir + '/MSMT17_V1/test'
        subprocess.Popen(cmd.split())
    elif dataset is None:
        pass
    else:
        raise Exception('unexpected dataset {}'.format(dataset))


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        
        # im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
        # exts = [args.image_ext]
        print(args.im_or_folder)
        # im_list = [p for p in glob.glob(os.path.join(args.im_or_folder, '**', '*'))
        #       if any(p.endswith(ext) for ext in exts)]
        # im_list = find_files(args.im_or_folder, ext = '*.' + args.image_ext )
        im_list = prutils.find_files(args.im_or_folder, ext='*.jpg')
    else:
        im_list = [args.im_or_folder]

    make_output_dirs(args.output_dir, args.im_or_folder, args.reid_dataset)

    logger.info(im_list)
    logger.info("-----The above is \"im_list\" variable ------------------------------------")
    # exit()
    for i, im_name in enumerate(im_list): 
        # if i > 2:
        #     logger.info('debug!')
        #     exit()

        extended_output_dir = make_output_dir_path(args.output_dir, im_name, args.reid_dataset)
        out_name = os.path.join(
            extended_output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        
        if os.path.exists(out_name):
            logger.info('Skip existing {}'.format(out_name))
            continue
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            extended_output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )
    


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
