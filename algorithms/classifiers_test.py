import numpy as np
import random as rd

import algorithms.clustering.KMeansClusteringClassifier as kMeans
import algorithms.classification.EnsembleLinearClassifier as ensemble
import algorithms.classification.MeanSquareErrorMinimizerLinearClassifier as linear


def run_KMeansClusteringClassifier():
    print('****************************************************************')
    print('K-Means Clustering Classifier')
    print('****************************************************************')
    for k in range(1, 10):
        print('num of classes:', k)
        classifier = kMeans.KMeansClusteringClassifier()
        successive_means, n_iterations = classifier.train(nd_data, k)
        print('classifier created in [%s] iterations' % n_iterations)
        # print('classifier:\n', classifier.classifier)
        # print('successive means:\n', successive_means)

        classification_output = classifier.classify(v_query)
        print('query %s classified as "%s"\n' % (v_query, classification_output))


def run_LinearClassifier():
    print('****************************************************************')
    print('Mean Square Error Minimizer - Linear Classifier')
    print('****************************************************************')
    classifier = linear.MeanSquareErrorMinimizerLinearClassifier()
    classifier.train(nd_data, target)
    classification_output = classifier.classify(v_query)
    print('query %s classified as "%s"\n' % (v_query, classification_output))


def run_EnsembleClassifier():
    print('****************************************************************')
    print('Ensemble - Linear Classifier')
    print('****************************************************************')
    for k in range(1, 10):
        print('num of linear classifiers:', k)
        classifier = ensemble.EnsembleLinearClassifier()
        classifier.train(nd_data, target, k)
        classification_output = classifier.classify(v_query)
        print('query %s classified as "%s"\n' % (v_query, classification_output))


if __name__ == "__main__":
    nd_data = (np.random.random_sample(1000) + np.random.random_sample(1000)).reshape(500, 2)
    target = [rd.randint(0, 2) for _ in range(500)]
    v_query = np.random.random_sample(2)
    print('random data to classify:\n', nd_data)
    print('----------------------------------------------------------------\n\n')
    run_KMeansClusteringClassifier()
    run_LinearClassifier()
    run_EnsembleClassifier()
