import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from decimal import Decimal, getcontext
from scipy.signal import argrelextrema

import pylab as pl

mpl.use('TkAgg')

getcontext().prec = 300


#Takes in alpha,a and the x point to create the y value of a sigmoid function
def sigmoidPoint(alpha, a, x):
    return 1 / (1 + np.exp(-alpha * (x - a)))


#Takes in an array of sigmoid finctions parameters [alpha, a, xmin, xmax] and returns an array of sigmoid functions x and y coordinates
def sigmoidArray(sigmoidparams):
    results = []
    sigmoidparams = sigmoidparams.astype(str)
    for sigmoid in sigmoidparams:
        alpha, a, xmin, xmax = map(Decimal, sigmoid)
        xcorr = np.arange(xmin, xmax + 1)
        ycorr = sigmoidPoint(alpha, a, xcorr)
        ycorr = alphaTransform(ycorr, alpha)
        results.append([xcorr, ycorr])
    return np.array(results)


#Transforms a point of a sigmoid function into a 1-0.5 , 0.5-0 form
def alphaTransform(ycorrs, alpha):
    if alpha < 0:
        ycorrs /= Decimal(2)
    elif alpha > 0:
        ycorrs = ycorrs / Decimal(2) + Decimal(0.5)
    return ycorrs


#Takes in an array of sigmoid functions (x and y coordinates)
# outputs an aggregated functions x and y coordinates
def sigmoidAggregate(sigmoidTomb):
    sigmoidTomb = np.array(sigmoidTomb)
    #print(sigmoidTomb)
    #yTomb = np.array([sigmoidTomb[1]])
    yTomb = np.array([sigmoid[1] for sigmoid in sigmoidTomb])

    preprod = (Decimal(1)-yTomb)/yTomb

    prod = np.prod(preprod, axis=0)

    yAggregated = Decimal(1) / (Decimal(1) + prod)

    xAggregated = sigmoidTomb[0][0]
    return np.array([xAggregated, yAggregated])


#Show the aggregate of an array of sigmoid functions
def aggregateShow(aggregateArray):
    plt.figure().set_figheight(2.5)
    aggregated = sigmoidAggregate(aggregateArray)

    plt.plot(aggregated[0], aggregated[1])
    plt.grid()
    plt.show()

#Show an array of sigmoid functions
def sigmoidArrayShow(sigmoidArray):
    plt.figure().set_figheight(2.5)
    for pliant in sigmoidArray:
        plt.plot(pliant[0], pliant[1])
    plt.grid()
    plt.ticklabel_format(style='plain')
    plt.show()

#Show an array of sigmoid functions and their aggregate
def sigmoidArrayAndAggregatePlot(sigmoidArray):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 7)

    aggregated = sigmoidAggregate(sigmoidArray)
    print(aggregated[1])
    ax1.plot(aggregated[0], aggregated[1])
    ax1.plot(0, 0)
    ax1.plot(0, 1)

    ax1.set_title("Aggregated")
    ax1.grid()

    for pliant in sigmoidArray:
        #print(pliant[1])
        ax2.plot(pliant[0], pliant[1])
    ax2.set_title("Sigmoid functions")

    plt.ticklabel_format(style='plain')
    ax2.grid()
    plt.show()


def createParamsArray(alpha, a, xmin, xmax):
    return np.array([[alpha, a, xmin, xmax]])


def appendParams(paramArray, alpha, a, xmin, xmax):
    paramArray = np.append(paramArray, [[alpha, a, xmin, xmax]], axis=0)
    return paramArray


#Takes a 2d array as parameter that consists of two arrays.
#The first array is x values, the second is y values
def aggMaxIndexes(aggregated):
    return np.array(argrelextrema(aggregated[1], np.greater, mode='wrap')[0])


#Takes a 2d array as parameter that consists of two arrays.
#The first array is x values, the second is y values
def aggMaxValues(aggregated):
    return aggregated[1][aggMaxIndexes(aggregated)]


#Takes a 2d array as parameter that consists of two arrays.
#The first array is x values, the second is y values
def aggMinIndexes(aggregated):
    return np.array(argrelextrema(aggregated[1], np.less, mode='wrap')[0])


#Takes a 2d array as parameter that consists of two arrays.
#The first array is x values, the second is y values
def aggMinValues(aggregated):
    return aggregated[1][aggMinIndexes(aggregated)]


def prayingToGod():
    pass

