import matplotlib.pyplot as plt
import numpy as np
import plotting.utils_plotting


def plot_predictions_confidence(
    truth,
    prediction_probability,
    bins=10,
    x_label="raining",
    y_label="preds_raining",
    filename=None,
):
    H, xedges, yedges = np.histogram2d(truth, prediction_probability, bins=(2, bins))
    fig = plt.figure(figsize=[10, 10])
    ax = plt.gca()
    pc = ax.pcolormesh(xedges, yedges, H.T, cmap="rainbow", shading="flat")
    plt.colorbar(pc)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plotting.utils_plotting.overplot_values(H, ax, 2, bins)
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")


def classification_report(labels, predictions, target_names=None, output_dict=True):
    import sklearn

    if target_names is None:
        target_names = ["not raining", "raining"]
    report = sklearn.metrics.classification_report(
        labels, predictions, target_names=target_names, output_dict=output_dict
    )
    return report


def check_prediction(truth, prediction, filename=None, normalize="all", output_dict=False):
    import sklearn

    report = classification_report(
        truth,
        prediction,
        target_names=["not raining", "raining"],
        output_dict=output_dict,
    )
    cm = sklearn.metrics.confusion_matrix(truth, prediction, normalize=normalize)
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not raining", "raining"])
    disp.plot()
    ax = plt.gca()
    ax.tick_params(axis="x", labelrotation=0)
    if filename is not None:
        figure = plt.gcf()
        figure.savefig(filename)
    return report


def plot_roc(truth, prediction_probability, filename=None):
    import sklearn.metrics

    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(
        truth, prediction_probability, drop_intermediate=False
    )
    roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    figure = plt.figure()
    lw = 2
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        lw=lw,
        label="ROC curve (area = %.04f)" % roc_auc,
    )

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()
    if filename is not None:
        figure.savefig(filename)
