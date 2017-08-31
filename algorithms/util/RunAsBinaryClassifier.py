import numpy as np
import algorithms.util.keslerization as ks


class RunAsBinaryClassifier:

    def __init__(self):
        self.unique_classes = None
        self.classifiers = []

    @staticmethod
    def getKeslerizedTarget(target, negative_value=-1):
        return ks.keslerize_column(target, negative_value)

    def createClassifiers(self, classifierType, nd_data, v_target, paramsForClassifierInit, *paramsForClassifierTraining):
        keslerized_columns, self.unique_classes = RunAsBinaryClassifier.getKeslerizedTarget(v_target)

        for k in range(len(self.unique_classes)):
            column = keslerized_columns[:, k]
            classifier = classifierType(*paramsForClassifierInit)
            classifier.train(nd_data, column, *paramsForClassifierTraining)
            self.classifiers.append(classifier)

    def runClassifiers(self, v_query):
        classification_output = []
        for k in range(len(self.unique_classes)):
            classification_output.append(self.classifiers[k].classify(v_query))
        return ks.de_keslerize_columns(np.array([classification_output]), self.unique_classes)
