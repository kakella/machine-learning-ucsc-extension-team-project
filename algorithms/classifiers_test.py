import numpy as np
import random as rd

import algorithms.classification.MeanSquareErrorMinimizerLinearClassifier as linear
import algorithms.classification.EnsembleLinearClassifier as ensemble

import algorithms.clustering.KMeansClusteringClassifier as kMeans
import algorithms.clustering.ExpectationMaximizationClassifier as expMax

import algorithms.util.RunAsBinaryClassifier as run

# import fma.FmaDataLoader as fma
import fma.excel_operations as eo


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
    for k in range(1000, 1001):
        print('num of linear classifiers:', k)
        runner = run.RunAsBinaryClassifier()
        classification_output = runner.runClassifier(ensemble.EnsembleLinearClassifier, nd_data, v_target, v_query, [],
                                                     k)
        print('query %s classified as "%s"\n' % (v_query, classification_output))


def run_ExpectationMaximizationClassifier():
    print('****************************************************************')
    print('Expectation Maximization Clustering Classifier')
    print('****************************************************************')
    for k in range(1, 10):
        classifier = expMax.ExpectationMaximizationClassifier(nd_data, k)
        classifier.Run()
        # classification_output = classifier.classify(v_query)
        # print('query %s classified as "%s"\n' % (v_query, classification_output))


def get_random_data():
    X = (np.random.random_sample(10000) + np.random.random_sample(10000)).reshape(500, 20)
    T = [rd.randint(0, 2) for _ in range(500)]
    Q = np.random.random_sample(35)
    return X, T, Q


def get_audio_data(feature):
    if feature == 'mfcc':
        inputExcelFile = r"../fma/data/mfcc_pca_dataset.xlsx"
        data = eo.readExcel(inputExcelFile)
    elif feature == 'tonnetz':
        inputExcelFile = r"../fma/data/tonnetz_pca_dataset.xlsx"
        data = eo.readExcel(inputExcelFile)
    else:
        return None, None

    inputExcelFile = r"../fma/data/small_target_dataset.xlsx"
    target = eo.readExcel(inputExcelFile)

    ###########################################################################
    # Below code only needs to be used once to create small_target_dataset.xlsx
    ###########################################################################
    # fmaObj = fma.FmaDataLoader('../fma/data')
    # _, target = fmaObj.load_specific_data(fmaObj.SUBSETS[0], feature)
    # outputExcelFile = r"../fma/data/small_target_dataset.xlsx"
    # eo.writeExcelData(data=[target],
    #                   excelFile=outputExcelFile,
    #                   sheetName='Sheet1',
    #                   startRow=2,
    #                   startCol=1)

    return data, target


if __name__ == "__main__":
    # nd_data, v_target, v_query = get_random_data()
    nd_data, v_target = get_audio_data('mfcc')
    v_target = v_target.T[0]

    print('data to classify:\n', nd_data.shape)
    print(nd_data)
    print('----------------------------------------------------------------\n\n')
    print('targets:\n', v_target.shape)
    print(v_target)
    print('----------------------------------------------------------------\n\n')
    # run_KMeansClusteringClassifier()
    # run_LinearClassifier()
    run_EnsembleClassifier()
    # run_ExpectationMaximizationClassifier()
