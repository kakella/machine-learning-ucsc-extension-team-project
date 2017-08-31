#! /usr/bin/env python3
#
# Usage:
#
# import Plotter
#
# Plotter.Plotter.PlotData(data_points, labels)
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plotter:

    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

    @classmethod
    def PlotData(cls, data_points, labels, title="title", xlabel="xlabel", ylabel="ylabel"):
        """
            data_points - Nx2 matrix of data points
            labels - Nx1 vector of string labels
            genres - tuple of 
        """

        plots = []
        color_index = 0
        unique_labels = np.unique(labels)
        for label in unique_labels:
            points = data_points[labels==label,:]
            c = Plotter.colors[color_index]
            plot = plt.scatter(points[:,0], points[:,1], marker='o', color=c)
            color_index += 1
            plots.append(plot)
        plt.legend(plots, unique_labels)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    @classmethod
    def PlotData3D(cls, data_points, labels, title="title", xlabel="xlabel", ylabel="ylabel"):
        """
            data_points - Nx3 matrix of data points
            labels - Nx1 vector of string labels
            genres - tuple of 
        """

        fig = plt.figure()
        #ax = fig.add_subplot(111, project='3d')            # New method after 1.0.0
        ax = Axes3D(fig)                                    # Old method before 1.0.0

        plots = []
        color_index = 0
        unique_labels = np.unique(labels)
        for label in unique_labels:
            points = data_points[labels==label,:]
            c = Plotter.colors[color_index]
            plot = ax.scatter(points[:,0], points[:,1], points[:,2], marker='o', color=c)
            color_index += 1
            plots.append(plot)
        plt.legend(plots, unique_labels)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    @classmethod
    def PlotPoint(cls, x, y, c):
        """No need to use this function, just call plt.scatter directly"""
        plt.scatter(x, y, marker='o', color=c)


