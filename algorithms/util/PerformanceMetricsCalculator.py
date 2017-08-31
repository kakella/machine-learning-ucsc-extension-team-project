import numpy as np
import algorithms.util.constants as cst


class PerformanceMetricsCalculator:

    def __init__(self):
        None

    @staticmethod
    def evaluate_binary_classifier_result(d_result, true_class_label, positive_class_label, negative_class_label):
        total = sum(d_result.values())
        if total == 0:
            return cst.INDETERMINATE_VALUE
        else:
            max_result = max(d_result.values())
            for r in d_result.items():
                if r[1] == max_result:
                    predicted_class_label = r[0]
                    if true_class_label == predicted_class_label:
                        if predicted_class_label == positive_class_label:
                            return 'TP'
                        else:
                            return 'TN'
                    else:
                        if predicted_class_label == positive_class_label:
                            return 'FP'
                        else:
                            return 'FN'

    @staticmethod
    def evaluate_binary_classifier(v_truth, v_result, positive_class_label, negative_class_label):
        TP = TN = FP = FN = 0
        for i, r in enumerate(v_result):
            if r == v_truth[i]:
                if r == positive_class_label:
                    TP += 1
                else:
                    TN += 1
            else:
                if r == positive_class_label:
                    FP += 1
                else:
                    FN += 1
        return {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        }


def binary_classifier_performance_metrics(classifier_output):
    return {
        'accuracy': (classifier_output['TP'] + classifier_output['TN']) /
                    (classifier_output['TP'] + classifier_output['TN'] +
                     classifier_output['FP'] + classifier_output['FN']),
        'sensitivity': classifier_output['TP'] /
                       (classifier_output['TP'] + classifier_output['FN']),
        'specificity': classifier_output['TN'] /
                       (classifier_output['FP'] + classifier_output['TN']),
        'positive_predictive_value': classifier_output['TP'] /
                                     (classifier_output['FP'] + classifier_output['TP']),
        'negative_predictive_value': classifier_output['TN'] /
                                     (classifier_output['FN'] + classifier_output['TN'])
    }


def evaluate_multiclass_classifier(truth, result):
    class_labels = sorted(set(truth))
    matrix_size = len(class_labels)
    confusion_matrix = np.zeros((matrix_size, matrix_size))

    for i, r in enumerate(result):
        truth_index = class_labels.index(truth[i])
        classified_index = class_labels.index(r)
        confusion_matrix[truth_index][classified_index] += 1

    return class_labels, confusion_matrix, multiclass_classifier_performance_metrics(confusion_matrix)


def multiclass_classifier_performance_metrics(confusion_matrix):
    matrix_size = len(confusion_matrix)
    metric_parameters = [
        {
            'TP': confusion_matrix[i][i],
            'TN': np.sum(np.delete(np.delete(confusion_matrix, (i), axis=0), (i), axis=1)),
            'FP': np.sum(confusion_matrix[:, i]) - confusion_matrix[i][i],
            'FN': np.sum(confusion_matrix[i, :]) - confusion_matrix[i][i]
        }
        for i in range(matrix_size)
    ]

    print('metric_parameters:\n', metric_parameters)

    performance = []

    for d in metric_parameters:
        if d['TP'] + d['TN'] + d['FP'] + d['FN'] > 0:
            accuracy = (d['TP'] + d['TN']) / (d['TP'] + d['TN'] + d['FP'] + d['FN'])
        else:
            accuracy = 0

        if d['TP'] + d['FN'] > 0:
            sensitivity = d['TP'] / (d['TP'] + d['FN'])
        else:
            sensitivity = 0

        if d['TN'] + d['FP'] > 0:
            specificity = d['TN'] / (d['TN'] + d['FP'])
        else:
            specificity = 0

        if d['FP'] + d['TP'] > 0:
            positive_predictive_value = d['TP'] / (d['FP'] + d['TP'])
        else:
            positive_predictive_value = 0

        if d['FN'] + d['TN'] > 0:
            negative_predictive_value = d['TN'] / (d['FN'] + d['TN'])
        else:
            negative_predictive_value = 0

        performance.append({
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': positive_predictive_value,
            'negative_predictive_value': negative_predictive_value
        })

    return performance


def accuracy(classifier_output):
    return {
        'bayesian': (classifier_output['bayesian']['TP'] + classifier_output['bayesian']['TN']) /
                    (classifier_output['bayesian']['TP'] + classifier_output['bayesian']['TN'] +
                     classifier_output['bayesian']['FP'] + classifier_output['bayesian']['FN']),
        'histogram': (classifier_output['histogram']['TP'] + classifier_output['histogram']['TN']) /
                     (classifier_output['histogram']['TP'] + classifier_output['histogram']['TN'] +
                      classifier_output['histogram']['FP'] + classifier_output['histogram']['FN'])
    }


def sensitivity(classifier_output):
    return {
        'bayesian': classifier_output['bayesian']['TP'] /
                    (classifier_output['bayesian']['TP'] + classifier_output['bayesian']['FN']),
        'histogram': classifier_output['histogram']['TP'] /
                     (classifier_output['histogram']['TP'] + classifier_output['histogram']['FN'])
    }


def specificity(classifier_output):
    return {
        'bayesian': classifier_output['bayesian']['TN'] /
                    (classifier_output['bayesian']['FP'] + classifier_output['bayesian']['TN']),
        'histogram': classifier_output['histogram']['TN'] /
                     (classifier_output['histogram']['FP'] + classifier_output['histogram']['TN'])
    }


def positive_predictive_value(classifier_output):
    return {
        'bayesian': classifier_output['bayesian']['TP'] /
                    (classifier_output['bayesian']['FP'] + classifier_output['bayesian']['TP']),
        'histogram': classifier_output['histogram']['TP'] /
                     (classifier_output['histogram']['FP'] + classifier_output['histogram']['TP'])
    }


def negative_predictive_value(classifier_output):
    return {
        'bayesian': classifier_output['bayesian']['TN'] /
                    (classifier_output['bayesian']['FN'] + classifier_output['bayesian']['TN']),
        'histogram': classifier_output['histogram']['TN'] /
                     (classifier_output['histogram']['FN'] + classifier_output['histogram']['TN'])
    }
