conda envs:
py2_dp_dljy    for running densepose.        (stage 1)
torch_py27     for running person reid nets. (stage 2)


$ ln -s /data/IUV-densepose/MSMT17_V1 ./person_reid_data/MSMT17_V1       (I've done this to bottleneck)
person_reid_data/ holds IUV, INDS, bbox data. 


/data/DensePose_ResNet101_FPN_s1x-e2e.pkl    cfg model file for densepose. Kept here instead of downloading it every time.

tools/infer_for_reid.py is a script for processing person reid datasets (person chips jpg) into IUV, INDS, bbox data to be stored in person_reid_data/ dir.
To process MSMT17_V1:
$ python tools/infer_for_reid.py --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml --output-dir ./person_reid_data --image-ext jpg --wts /data/DensePose_ResNet101_FPN_s1x-e2e.pkl --throttle=0.5 /data/MSMT17_V1 
Outputs to /person_reid_data/MSMT17_V1/train/0000/{________IUV.png, ______INDS.png, ________.pdf}

example-siamese.py  Example siamese ent.

example-basic-train.py  Example basic training.

test.py       For testing net.

resnet_custom.py
A custom resnet that can take in IUV stacks as input.

person_reid_nets.py
Nets for person reid. E.g. siamese-ize a net.

person_reid_utils.py
codes that help in person reid.

IUV_stack_utils.py
Codes for manipulating IUV stacks.

msmt17_v1_utils.py
Codes for dealing with msmt17_v1 dataset.

