import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl

mpl.use('TkAgg')

#Takes in alpha,a and the x point to create the y value of a sigmoid function
def sigmoidPoint(alpha, a, x):
    return 1 / (1 + np.exp(-alpha * (x - a)))

#Takes in an array of sigmoid finctions parameters [alpha, a, xmin, xmax] and returns an array of sigmoid functions x and y coordinates
def sigmoidArray(sigmoidparams):
    results = []
    for sigmoid in sigmoidparams:
        alpha, a, xmin, xmax = sigmoid
        xcorr = np.arange(xmin, xmax + 1)
        ycorr = sigmoidPoint(alpha, a, xcorr)
        ycorr = alphaTransform(ycorr, alpha)
        results.append([xcorr, ycorr])
    return np.array(results)

#Transforms a point of a sigmoid function into a 1-0.5 , 0.5-0 form
def alphaTransform(ycorrs, alpha):
    if alpha < 0:
        ycorrs /= 2
    elif alpha > 0:
        ycorrs = ycorrs / 2 + 0.5
    return ycorrs

#Takes in an array of sigmoid functions (x and y coordinates)
# outputs an aggregated functions x and y coordinates
def sigmoidAggregate(sigmoidTomb):
    sigmoidTomb = np.array(sigmoidTomb)
    yTomb = np.array([sigmoid[1] for sigmoid in sigmoidTomb])

    preprod = (1-yTomb)/yTomb

    prod = np.prod(preprod, axis=0)

    yAggregated = 1/(1+prod)
    xAggregated = sigmoidTomb[0][0]
    return np.array([xAggregated, yAggregated])

def aggregateShow(aggregateArray):
    plt.figure().set_figheight(2.5)
    plt.plot(aggregateArray[0], aggregateArray[1])
    plt.grid()
    plt.show()

def sigmoidArrayShow(sigmoidArray):
    plt.figure().set_figheight(2.5)
    for pliant in sigmoidArray:
        plt.plot(pliant[0], pliant[1])
    plt.grid()
    plt.show()

def sigmoidArrayAndAggregatePlot(sigmoidArray):
    plt.figure().set_figheight(2.5)
    aggregated = sigmoidAggregate(sigmoidArray)

    plt.subplot(1, 2, 1)
    plt.plot(aggregated[0], aggregated[1])
    plt.title("Aggregated")
    plt.grid()

    plt.subplot(1, 2, 2)
    for pliant in sigmoidArray:
        plt.plot(pliant[0], pliant[1])
    plt.title("Sigmoid functions")

    pl.grid()
    plt.show()


plotmin = 0
plotmax = 50


params = np.array([[-1, 7, plotmin, plotmax]])

for i in range(3):
    params = np.append(params, [[random.uniform(-3, 3), random.randint(7, 40), plotmin, plotmax]], axis=0)


pliantok = sigmoidArray(params)
sigmoidArrayAndAggregatePlot(pliantok)

