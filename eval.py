import scipy 
import numpy as np
import h5py
from sklearn.metrics import f1_score

def main():
    path = 'summe/GT/'
    file = 'Air_Force_One.mat'
    gt_data = scipy.io.loadmat(path + file)
    user_score = gt_data.get('user_score')
    print(user_score.shape)
    nFrames = user_score.shape[0]
    nbOfUsers = user_score.shape[1]
    
    path = 'result/'
    machine_binary_vector = np.zeros((nFrames, 1))
    file = 'Air_Force_One.h5'
    with h5py.File(path + file, 'r') as hf:
        summary_selection = hf['video_0/machine_summary'][:]
        print(summary_selection.shape)
        machine_binary_vector[summary_selection > 0] = 1
    
    recall = np.zeros((nbOfUsers, 1))
    precision = np.zeros((nbOfUsers, 1))
    for userIdx in range(0, nbOfUsers):
        gt_binary_vector = np.zeros((nFrames, 1))
        gt_binary_vector[user_score[:, userIdx] > 0] = 1
        recall[userIdx] = np.sum(gt_binary_vector * machine_binary_vector) / np.sum(gt_binary_vector)
        precision[userIdx] = np.sum(gt_binary_vector * machine_binary_vector) / np.sum(machine_binary_vector)
    f_measure = np.zeros((nbOfUsers, 1))
    for idx in range(0, len(precision)):
        if precision[idx] > 0 or recall[idx] > 0:
            f_measure[idx] = 2 * recall[idx] * precision[idx] / (recall[idx] + precision[idx])
        else:
            f_measure[idx] = 0
    f_measure = np.mean(f_measure)
    print(f"Recall: {np.mean(recall):.3f}")
    print(f"Precision: {np.mean(precision):.3f}")
    print(f"F-measure: {f_measure:.3f}")
    
if __name__ == "__main__":
    main() 