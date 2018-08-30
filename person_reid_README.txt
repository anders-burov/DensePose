person_reid_data/ holds IUV, INDS, bbox data.

/data/DensePose_ResNet101_FPN_s1x-e2e.pkl    cfg model file for densepose. Kept here instead of downloading it every time.

infer_for_reid.py is a script for processing person reid datasets (person chips jpg) into IUV, INDS, bbox data to be stored in person_reid_data/ dir.
To process MSMT17_V1:
$ python tools/infer_for_reid.py --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml --output-dir ./person_reid_data --image-ext jpg --wts /data/DensePose_ResNet101_FPN_s1x-e2e.pkl --throttle=1.0 /data/MSMT17_V1 
Outputs to /person_reid_data/MSMT17_V1/train/0000/{________IUV.png, ______INDS.png, ________.pdf}