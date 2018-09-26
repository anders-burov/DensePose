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

precompute_IUV_stacks_MSMT17.py   Precomputes IUV stacks on test dataset. Precomputation is for preventing OOM error when performing inference on whole test dataset.
Currently saved at /data/IUV-densepose/MSMT17_V1/precomputed

example-siamese.py  Example siamese ent.

example_basic_train.py  Example basic training. There's a newer training script e.g. train_msmt17.py

train_msmt17.py         Training.
E.g. of overfit:
$ python train_msmt17.py --device="cuda:3" --num_worker=2
E.g. of real train:
$ python train_msmt17.py --odir="expt-32-twomarg-0.1-0.7-nomask" --device="cuda:3" --batch_size 32 --num_worker 8 --overfit 0

test_MSMT17_v2.py       
For testing net.
$ python test_MSMT17_v2.py --net="expt-32-twomarg-0.1-0.7/net-ep-40.chkpt" --pos_margin=0.1 --neg_margin=0.7 --odir="tmp/p0.1n0.1ep40" --n_probes=50 --n_gallery=450 --batch_size=25 --num_worker=8

Compute ROC, DET:
$ pip install pyeer
$ python investigate_similarity_and_intersection.py    #RUNS FAST. NEEDS PRECOMPUTED STUFF FROM test_MSMT17_v2.py
$ geteerinf -p "tmp/p0.1n0.7ep80/ROC" -i "imposter-nofilt.txt" -g "genuine-nofilt.txt" -lw 1
$ geteerinf -p "tmp/p0.1n0.7ep80/ROC" -i "imposter-filtmedian.txt" -g "genuine-filtmedian.txt" -e "filtmedian" -lw 1 -sp "tmp/p0.1n0.7ep80/ROC" -s
To do multiple genuine & imposter score pairs at once:
$ geteerinf -p "tmp/p0.1n0.7ep80/ROC"      -i "imposter-above_0_percentile.txt,imposter-above_10_percentile.txt,imposter-above_20_percentile.txt,imposter-above_30_percentile.txt,imposter-above_40_percentile.txt,imposter-above_50_percentile.txt"      -g "genuine-above_0_percentile.txt,genuine-above_10_percentile.txt,genuine-above_20_percentile.txt,genuine-above_30_percentile.txt,genuine-above_40_percentile.txt,genuine-above_50_percentile.txt"        -lw 1 -ls 10 -sp "tmp/p0.1n0.7ep80/ROC" -s


resnet_custom.py
A custom resnet that can take in IUV stacks as input.

person_reid_nets.py
Nets for person reid. E.g. siamese-ize a net.

person_reid_utils.py
codes that help in person reid.

IUV_stack_utils.py
Codes for manipulating IUV stacks.

loss_and_metrics.py
Losses, metrics, and plotting code.

msmt17_v1_utils.py
Codes for dealing with msmt17 dataset.
Codes for msmt17 pytorch Dataset class.

