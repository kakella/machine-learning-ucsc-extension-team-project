import numpy as np
import algorithms.util.keslerization as ks


class RunAsBinaryClassifier:

    def __init__(self):
        None

    @staticmethod
    def getKeslerizedTarget(target, negative_value=-1):
        return ks.keslerize_column(target, negative_value)

    @staticmethod
    def runClassifier(classifierType, nd_data, v_target, v_query, paramsForClassifierInit, *paramsForClassifierTraining):
        keslerized_columns, unique_classes = RunAsBinaryClassifier.getKeslerizedTarget(v_target)

        classification_output = []
        for k in range(len(unique_classes)):
            column = keslerized_columns[:, k]
            classifier = classifierType(*paramsForClassifierInit)
            classifier.train(nd_data, column, *paramsForClassifierTraining)
            classification_output.append(classifier.classify(v_query))

        return ks.de_keslerize_columns(np.array([classification_output]), unique_classes)

