import numpy as np
import random as rd

import algorithms.classification.MeanSquareErrorMinimizerLinearClassifier as linear
import algorithms.classification.EnsembleLinearClassifier as ensemble

import algorithms.clustering.KMeansClusteringClassifier as kMeans
import algorithms.clustering.EM as expMax

import algorithms.util.RunAsBinaryClassifier as run


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
    classifier.train(nd_data, v_target)
    classification_output = classifier.classify(v_query)
    print('query %s classified as "%s"\n' % (v_query, classification_output))


def run_EnsembleClassifier():
    print('****************************************************************')
    print('Ensemble - Linear Classifier')
    print('****************************************************************')
    for k in range(100, 101):
        print('num of linear classifiers:', k)
        runner = run.RunAsBinaryClassifier()
        classification_output = runner.runClassifier(ensemble.EnsembleLinearClassifier, nd_data, v_target, v_query, [], k)
        print('query %s classified as "%s"\n' % (v_query, classification_output))


def run_ExpectationMaximizationClassifier():
    print('****************************************************************')
    print('Expectation Maximization Clustering Classifier')
    print('****************************************************************')
    for k in range(1, 10):
        classifier = expMax.EM(nd_data, k)
        classifier.Run()
        # classification_output = classifier.classify(v_query)
        # print('query %s classified as "%s"\n' % (v_query, classification_output))


def get_data():
    X = (np.random.random_sample(1000) + np.random.random_sample(1000)).reshape(500, 2)
    T = [rd.randint(0, 2) for _ in range(500)]
    Q = np.random.random_sample(2)
    return X, T, Q


if __name__ == "__main__":
    nd_data, v_target, v_query = get_data()
    print('data to classify:\n', nd_data)
    print('----------------------------------------------------------------\n\n')
    # run_KMeansClusteringClassifier()
    # run_LinearClassifier()
    run_EnsembleClassifier()
    # run_ExpectationMaximizationClassifier()
