import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay


def save_precision_recall_curve(anomaly_scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, anomaly_scores)
    average_precision = average_precision_score(labels, anomaly_scores)
    PrecisionRecallDisplay(precision, recall, average_precision=average_precision, estimator_name='CAE').plot()
    plt.show()
    plt.savefig('precision_recall_curve.png', dpi=400)
    return precision, recall, thresholds
