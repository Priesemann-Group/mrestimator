import neo, pickle
import numpy as np

"""
This module was only used to generate the activity matrices in data for the tests.
"""

def make_activity_matrix(trials_iterator, min_len_trial, ele_num, max_num_trials = None):
    activity_mat = []
    for trial_counter, (segment, analogsignal) in enumerate(trials_iterator):
        if not max_num_trials is None and trial_counter >= max_num_trials:
            break
        if segment is None:
            activity = np.nan
        else:
            activity = np.squeeze(np.array(analogsignal[:min_len_trial, ele_num], dtype="float"))
        activity_mat.append(activity)

    return np.array(activity_mat)

def trial_iterator_type2(block):
    for i, segment in enumerate(block.segments):
        segment.index = i
        if segment.annotations["trial_type"] == 2 and len(segment.analogsignals) == 1:
            yield segment, segment.analogsignals[0] 

def get_min_len_trial(trials_iterator):
    min_len_trial = 1e42
    for segment, analogsignal in trials_iterator:
        if segment is None:
            continue
        len_trial = len(analogsignal[:, 0])
        if len_trial < min_len_trial:
            min_len_trial = len_trial
    if min_len_trial > 1e41:
        return None
    else:
        return min_len_trial

def save_activity():
    path_data = "/scratch.local/jdehning/141024/session01/data_neo_mua.pickled"
    file = neo.io.PickleIO(path_data)
    block = file.read_block()
    trials_iterator = trial_iterator_type2
    min_len_trial = get_min_len_trial(trials_iterator(block))
    if min_len_trial is None:
        return
    for ele_num in range(0,40,10):
        activity_mat = make_activity_matrix(trials_iterator(block), min_len_trial=min_len_trial,
                                            ele_num=ele_num)[:, 5:2005]
        pickle.dump(activity_mat, open("./data/activity_mat_{}.pickled".format(ele_num), "wb"))

if __name__ == "__main__":
    save_activity()

