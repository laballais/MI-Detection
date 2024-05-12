import numpy as np
from scipy import spatial
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = 0
    FN = 0
    TP = 0
    TN = 0

    # 0 - non MI : 1 - MI ---> concept 2
    for i in range(len(prediction)):
        if prediction[i] == 1  and groundtruth[i] == 0:
            FP += 1
        if prediction[i] == 0  and groundtruth[i] == 1:
            FN += 1
        if prediction[i] == 1  and groundtruth[i] == 1:
            TP += 1
        if prediction[i] == 0  and groundtruth[i] == 0:
            TN += 1

    # # 0 - MI : 1 - non MI ---> concept 1
    # for i in range(len(prediction)):
    #     if prediction[i] == 0  and groundtruth[i] == 1:
    #         FP += 1
    #     if prediction[i] == 1  and groundtruth[i] == 0:
    #         FN += 1
    #     if prediction[i] == 0  and groundtruth[i] == 0:
    #         TP += 1
    #     if prediction[i] == 1  and groundtruth[i] == 1:
    #         TN += 1
    
    return FP, FN, TP, TN

def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0

def sensitivity_score(prediction, groundtruth):
    """Getting the sensitivity of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    sensitivity = np.divide(TP, TP + FN + 1e-6)
    return sensitivity * 100.0

def specificity_score(prediction, groundtruth):
    """Getting the specificity of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    specificity = np.divide(TN, TN + FP + 1e-6)
    return specificity * 100.0

def precision_score(prediction, groundtruth):
    """Getting the specificity of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    precision = np.divide(TP, TP + FP + 1e-6)
    return precision * 100.0

def plot_confusionMatrix(prediction, groundtruth):
    cf_matrix = confusion_matrix(prediction, groundtruth)
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2', cmap='Blues')

    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Class')
    ax.set_ylabel('Actual Class')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Non MI','MI'])
    ax.yaxis.set_ticklabels(['Non MI','MI'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()