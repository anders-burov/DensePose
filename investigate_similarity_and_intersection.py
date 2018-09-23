from __future__ import print_function
import pickle
from cmc import count as cmc_count
import numpy as np
from datetime import datetime
import os

# Generates the scores of the false and true pairs.
# Removes scores that came about (unfortunately) with low intersection amounts.
# Later you can use these scores to compute ROC.

def get_genuine_and_imposter_scores(similarity_mat, intersection_mat, threshold):
    # args:
    #  E.g. threshold = 16000 means any score that is associated with an intersection amt
    #       of less than 16000 is NaN'ed and subsequently removed from the returned scores.
    assert(np.array_equal(np.array(similarity_mat.shape), np.array(intersection_mat.shape)))
    n_orig_gallery, n_orig_probes = similarity_mat.shape
    sim_mat = similarity_mat.copy()
    interx_mat = intersection_mat.copy()
    sim_mat[interx_mat < threshold] = np.nan
    genuine_scores = np.diag(sim_mat)
    n_genuine_total = len(np.diag(sim_mat))
    genuine_scores = genuine_scores[~np.isnan(genuine_scores)]
    print('genuine_scores size is', len(genuine_scores), 'out of orignilly ', n_genuine_total)
    mask_genuine = np.zeros_like(sim_mat) + \
                    np.concatenate([np.eye(n_genuine_total), np.zeros((n_orig_gallery-n_orig_probes,n_orig_probes))], axis=0)
    mask_imposter = (mask_genuine -1 ) * -1
    mask_imposter = mask_imposter.astype(np.int)
    # print(np.unique(mask_imposter))
    imposter_scores = sim_mat[mask_imposter > 0.5]
    # print(np.unique(imposter_scores))
    imposter_scores = imposter_scores[~np.isnan(imposter_scores)]
    # print(sim_mat.size, 'vs num of imposter scores:', imposter_scores.shape)
    return genuine_scores, imposter_scores


def main():
    expt_name = ''
    script_start_time = "{:%m-%d-%H-%M-%S}".format(datetime.now()); print('script_start_time ', script_start_time)
    idir = 'tmp/p0.1n0.7ep80'
    logbk_path = os.path.join(idir, 'test-logbk-09-23-17-48-06.pkl')
    odir = os.path.join(idir, 'ROC')
    if not os.path.exists(odir):
        os.makedirs(odir)
    logbk = pickle.load(open('tmp/p0.1n0.7ep80/test-logbk-09-23-17-48-06.pkl','rb'))
    sim_mat = logbk['similarity_mat']
    interx_mat = logbk['intersection_mat']
    for q in np.arange(0, 11, 10):
        thresh = np.percentile(interx_mat, q=q)
        print('At', q, '%:', thresh, 'intersection amt.')
        tmp_expt_name = expt_name + 'above_' + str(q) + '_percentile'
        genuine_scores, imposter_scores = get_genuine_and_imposter_scores(sim_mat.copy(), interx_mat.copy(), thresh)
        np.savetxt(os.path.join(odir, 'genuine-{}.txt'.format(tmp_expt_name)), genuine_scores)
        np.savetxt(os.path.join(odir, 'imposter-{}.txt'.format(tmp_expt_name)), imposter_scores)


    # exit()

    # # # print(np.diag(sim_mat).shape, np.diag(sim_mat))
    # # if remove_scores_below_median:
    # #     thres_interx = np.median(interx_mat)
    # # else:
    # #     thres_interx = 0
    # print('threshold intersction:', thres_interx)
    # genuine_scores, imposter_scores = get_genuine_and_imposter_scores(sim_mat, interx_mat, thres_interx)
    # exit()
    # np.savetxt(os.path.join(odir, 'genuine-{}.txt'.format(expt_name)), genuine_scores)
    # np.savetxt(os.path.join(odir, 'imposter-{}.txt'.format(expt_name)), imposter_scores)


if __name__ == "__main__":
    main()