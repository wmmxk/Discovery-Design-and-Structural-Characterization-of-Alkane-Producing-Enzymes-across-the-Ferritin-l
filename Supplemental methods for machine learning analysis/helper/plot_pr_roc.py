from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


def plot_pr_curve(truth, pred, save_path, title):
    plt.figure(figsize=(5, 3.5))
    precision, recall, _ = precision_recall_curve(truth, pred)
    average_precision = average_precision_score(truth, pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.8,where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title + 'average precision={0:0.2f}'.format(average_precision))
    plt.savefig(save_path)
    plt.close()
    return average_precision


def plot_roc_curve(truth, pred, save_path, title):
    fpr, tpr, _ = roc_curve(truth, pred)
    auc_roc = auc(fpr, tpr)

    plt.figure(figsize=(5, 3.5))
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + 'auc under roc = {0:0.2f}'.format(auc_roc))
    plt.savefig(save_path)
    plt.close()
    return auc_roc
