import numpy
import PIL.Image
import sklearn.metrics
import matplotlib.pyplot as plt

root = "M://Diploma/experiments/vesuvius/fragments"

def estimate_quality(pred_proba: numpy.ndarray, masks: numpy.ndarray) -> dict:
    scores = pred_proba.flatten()
    preds = (scores >= 0.5).astype(numpy.int8)
    targets = masks.flatten().astype(numpy.int8)
    return {
        "F0.5": sklearn.metrics.fbeta_score(targets, preds, beta=0.5, zero_division=0),
        "Accuracy": sklearn.metrics.accuracy_score(targets, preds),
        "AUC-ROC": sklearn.metrics.roc_auc_score(targets, scores),
        "Precision": sklearn.metrics.precision_score(targets, preds, zero_division=0),
        "Recall": sklearn.metrics.recall_score(targets, preds, zero_division=0),
        "F1-score": sklearn.metrics.f1_score(targets, preds, zero_division=0),
        "Jaccard score": sklearn.metrics.jaccard_score(targets, preds, zero_division=0),
    }

def plot_prediction(pred_proba: numpy.ndarray, fragment: int, ax, thr: float = 0.5):
    mask = numpy.array(PIL.Image.open(f"{root}/Frag{fragment}/mask.png")) > 0
    prediction = numpy.zeros(mask.shape, dtype=bool)
    prediction[mask] = pred_proba.ravel() >= thr

    if ax is None:
        target = numpy.array(PIL.Image.open(f"{root}/Frag{fragment}/inklabels.png")) > 0
        fig, axes = plt.subplots(1, 2, figsize=(4, 3))
        axes[0].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Ink target")
        axes[1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Predicted ink")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    else:
        ax.imshow(prediction, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Predicted ink")
        ax.axis("off")
