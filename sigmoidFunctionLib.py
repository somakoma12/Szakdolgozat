import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.signal import argrelextrema, find_peaks
from scipy.ndimage import gaussian_filter1d


mpl.use('TkAgg')



#Takes in alpha,a and the x point to create the y value of a sigmoid function
def sigmoidPoint(alpha, a, x):
    return 0.99 / (1 + np.exp(-alpha * (x - a)))


#Takes in an array of sigmoid finctions parameters [alpha, a, xmin, xmax] and returns an array of sigmoid functions x and y coordinates
def sigmoidArray(sigmoidparams):
    results = []
    for sigmoid in sigmoidparams:
        alpha, a, xmin, xmax, direction = sigmoid
        xcorr = np.arange(xmin, xmax + 1)
        ycorr = sigmoidPoint(alpha, a, xcorr)
        ycorr = alphaTransform(ycorr, direction)
        results.append([xcorr, ycorr])
    return np.array(results, dtype=np.float64)


#Transforms a point of a sigmoid function into a 1-0.5 , 0.5-0 form
def alphaTransform(ycorrs, direction):
    if direction:
        ycorrs = (1 + ycorrs) / 2
    else:
        ycorrs = (1 - ycorrs) / 2
    return ycorrs


#Takes in an array of sigmoid functions (x and y coordinates)
# outputs an aggregated functions x and y coordinates
def sigmoidAggregate(sigmoidTomb):
    sigmoidTomb = np.array(sigmoidTomb)
    #print(sigmoidTomb)
    #yTomb = np.array([sigmoidTomb[1]])
    yTomb = np.array([sigmoid[1] for sigmoid in sigmoidTomb])

    preprod = (1-yTomb)/yTomb

    prod = np.prod(preprod, axis=0)

    yAggregated = 1 / (1 + prod)

    xAggregated = sigmoidTomb[0][0]
    return np.array([xAggregated, yAggregated])


#Show the aggregate of an array of sigmoid functions
def aggregateShow(aggregateArray):
    plt.figure().set_size_inches(14, 7)

    plt.plot(aggregateArray[0], aggregateArray[1])
    plt.plot(aggregateArray[0][0], 0)
    plt.plot(aggregateArray[0][0], 1)


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


def createParamsArray(alpha, a, xmin, xmax, direction):
    return np.array([[alpha, a, xmin, xmax, direction]], dtype=np.float64)


def appendParams(paramArray, alpha, a, xmin, xmax, direction):
    paramArray = np.append(paramArray, [[alpha, a, xmin, xmax, direction]], axis=0)
    return paramArray


#Takes a 2d array as parameter that consists of two arrays.
#The first array is x values, the second is y values
def aggMaxIndexes(aggregated):
    return np.array(find_peaks(aggregated[1])[0])
    #return np.array(argrelextrema(aggregated[1], np.greater, mode='wrap')[0])


#Takes a 2d array as parameter that consists of two arrays.
#The first array is x values, the second is y values
def aggMaxValues(aggregated):
    return np.array(aggregated[1][aggMaxIndexes(aggregated)])


#Takes a 2d array as parameter that consists of two arrays.
#The first array is x values, the second is y values
def aggMinIndexes(aggregated):
    return np.array(find_peaks(-aggregated[1])[0])
    #return np.array(argrelextrema(aggregated[1], np.less, mode='wrap')[0])


#Takes a 2d array as parameter that consists of two arrays.
#The first array is x values, the second is y values
def aggMinValues(aggregated):
    return np.array(aggregated[1][aggMinIndexes(aggregated)])


def aggMinMaxIndexesSorted(aggregated):
    mins = aggMinIndexes(aggregated)
    maxes = aggMaxIndexes(aggregated)
    sortedindexes = None
    if mins.size == 0 and maxes.size == 0:
        if aggregated[1][0] == aggregated[1][-1]:
            return None
        else:
            return np.array([0, aggregated[1].size-1])

    sortedindexes = np.array([0, aggregated[1].size-1])
    sortedindexes = np.append(sortedindexes, mins)
    sortedindexes = np.append(sortedindexes, maxes)
    sortedindexes = np.sort(sortedindexes)

    return sortedindexes


# Computes the first sigmoid curve from the aggregate,
# based on its first inflection point and alpha parameter.
def firstSigmoidOfAggregateFromInflectionPointKnowingAlpha(aggregate, alpha):
    minmaxes = aggMinMaxIndexesSorted(aggregate)
    xmin = aggregate[0][0]
    xmax = aggregate[0][-1]
    aggregateY = np.array(aggregate[1])

    aggregateY[(0.499999 < aggregateY) & (aggregateY < 0.500001)] = 0.5
    inputaggregateY = gaussian_filter1d(aggregateY, 4)

    derrivates = np.gradient(np.gradient(inputaggregateY))
    infls = np.where(1 < abs(np.diff(np.sign(derrivates))))[0]

    subtractionParams = createParamsArray(alpha, infls[0] + xmin, xmin, xmax, minmaxes[0] < minmaxes[1])
    subtractionSigmoid = sigmoidArray(subtractionParams)

    return subtractionSigmoid


# Divides the aggregates prod by a sigmoid,
# to modify its values and returns the adjusted aggregate.
def subtractSigmoidFromAggregateByDivide(inputaggregate, sigmoid):
    returnaggregate = inputaggregate

    inputprod = (1 / inputaggregate[1]) - 1
    preprod = (1 - sigmoid[0][1]) / sigmoid[0][1]
    prod = inputprod / preprod

    returnaggregate[1] = inputaggregate[1] = 1 / (1 + prod)

    return returnaggregate


# Subtracts the first sigmoid from the aggregate by dividing its prod,
# based on the alpha parameter.
def subtractFirstSigmoidByDivideKnowingAlpha(inputaggregate, alpha):
    firstsigmoid = firstSigmoidOfAggregateFromInflectionPointKnowingAlpha(inputaggregate, alpha)

    newaggregate = subtractSigmoidFromAggregateByDivide(inputaggregate, firstsigmoid)

    return newaggregate


#Takes a 2d array as parameter that consists of two arrays.
#The first array is x values, the second is y values
def prayingToGod():
    pass

