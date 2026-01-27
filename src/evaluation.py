import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, title = "confusion matrix ", filename=None):
    """
    Plot and save the confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    labels : list
        List of class labels
    filename : str
        Path to save the confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()

def classification_report_text(y_true, y_pred):
    """
    Generate classification report text.

    Parameters
    ----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels

    Returns
    -------
    report : str
        Textual classification report
    """
    return classification_report(y_true, y_pred)

def plot_roc_curve(y_true, y_prob, title="ROC Curve", filename=None):

    """
    Plot and save the ROC curve.

    Parameters
    ----------
    y_true : array-like
        True class labels
    y_prob : array-like
        Predicted probabilities for the positive class
    title : str
        Title of the plot
    filename : str
        Path to save the ROC curve plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

    return auc 

def plot_roc_curve_comparison(y_true, prob_dict, filename):
    """
    prob_dict: dict of {model_name: y_prob}
    """
    plt.figure(figsize=(6,5))

    for model_name, y_prob in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")

    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show()