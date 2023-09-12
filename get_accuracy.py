import numpy as np
import scipy
import matplotlib.pyplot as plt

plot=False
def get_accuracies(expected, predicted):
    ccs = []
    for i, expected_text in enumerate(expected):
        predict_text = predicted[i]
        cc, expected_ranks, predicted_ranks = get_accuracy(expected_text, predict_text)
        ccs.append(cc)
    return np.nanmean(ccs)

def get_accuracy(expected_text, predict_text, plot=False):
    """Get accuracy of the predicted text."""
    expected_list = expected_text.split(' ')
    predict_list = predict_text.split(' ')
    expected = dict()
    predict = dict()

    genes = [f'G{g}' for g in range(1,101)]
    for i, gene in enumerate(genes):
        expected[gene] = np.nan
        predict[gene] = np.nan
        
    for i, gene in enumerate(expected_list):
        expected[gene] = i
    for i, gene in enumerate(predict_list):
        predict[gene] = i
    # calculate spearman rank correlation between expected and predict
    cc = scipy.stats.spearmanr(list(expected.values()), list(predict.values()), nan_policy='omit')
    expected_ranks = list(expected.values())
    predicted_ranks = list(predict.values())
    if plot:
        plt.figure(figsize=(4,4))
        plt.scatter(expected_ranks, predicted_ranks, s=1)
        plt.xlabel('target')
        plt.ylabel('initial or predict')
        plt.title('Rank Correlation')
        plt.show()
        plt.close()
    return cc[0], expected_ranks, predicted_ranks

    
        