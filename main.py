import sigmoidFunctionLib as sfl

import numpy as np
import random
#vannak
#iteration of divide first sigmoid = divideAggregateProdByFirstSigmoidKnowingAlpha
#iteration of divide first sigmoid = subtractFirstSigmoidFromAggregateKnowingAlpha
#all sigmoids of aggregate from local min-max points knowing alpha = allSigmoidsFromLocalValuesHalvingKnowingAlpha
#all sigmoids of aggregate from inflection points knowing alpha = allSigmoidsFromInflectionPointKnowingAlpha
#

plotmin = -100
plotmax = 500
#params setup
params = sfl.createParamsArray(1/4, 100, plotmin, plotmax, True)
params = sfl.appendParams(params, 1/4, 140, plotmin, plotmax, False)
params = sfl.appendParams(params, 1/4, 200, plotmin, plotmax, False)
params = sfl.appendParams(params, 1/4, 250, plotmin, plotmax, True)
print(params)
#slice first sigmoid
paramsminfirst = params[1:4]

#params into sigmoids
sigmoidok = sfl.sigmoidArray(params)
sigmoidokminfirst = sfl.sigmoidArray(paramsminfirst)

#sigmoids aggregated
aggregate = sfl.sigmoidAggregate(sigmoidok)
aggregateminfirst = sfl.sigmoidAggregate(sigmoidokminfirst)

#dividing aggregation prod by first sigmoid
divided = np.array(aggregate)
for i in range(1, 1):
    sfl.aggregateShow(divided)
    print("divide"+str(i))
    divided = sfl.divideAggregateProdByFirstSigmoidKnowingAlpha(divided, 1 / 4)

#subtracting sigmoid from aggregate
subtracted = np.array(aggregate)
for i in range(1, 1):
    sfl.aggregateShow(subtracted)
    print("subtracted"+str(i))
    subtracted = sfl.subtractFirstSigmoidFromAggregateKnowingAlpha(subtracted, 1 / 4)

#getting all sigmoids with half value
#halfmodesigmoids = sfl.allSigmoidsFromLocalValuesHalvingKnowingAlpha(aggregate, 1/4)
#halfmodeagg = sfl.sigmoidAggregate(halfmodesigmoids)


#getting all sigmoids with inflexion points
inflexsigmoids = sfl.allSigmoidsFromInflectionPointKnowingAlpha(aggregate, 1/4)
inflexagg = sfl.sigmoidAggregate(inflexsigmoids)


#quick hacks
#minmaxvalues = sfl.aggMinMaxValuesSortedByIndex(aggregate)
#minmaxindexes = sfl.aggMinMaxIndexesSorted(aggregate)
#halfvalue = (minmaxvalues[0]+minmaxvalues[1]) / 2
#index = sfl.findHalfBetweenLocalValues(aggregate, minmaxindexes[0], minmaxindexes[1], halfvalue)
#params = sfl.createParamsArray(1/4, index, plotmin, plotmax, True)
#sigmoid = sfl.sigmoidArray(params)
#aggregate = sfl.divideAggregateProdBySigmoid(aggregate, sigmoid)

#halfmodeagg[1] = aggregate[1] - halfmodeagg[1]



#plots
#sfl.sigmoidArrayShow(sigmoidok)
#sfl.sigmoidArrayShow(halfmodesigmoids)
#sfl.sigmoidArrayShow(inflexsigmoids)
#sfl.aggregateShow(aggregate)
#sfl.aggregateShow(divided)
#sfl.aggregateShow(halfmodeagg)
sfl.aggregateShow(inflexagg)
#sfl.aggregateShow(aggregateminfirst)
#sfl.aggregateShow(subtracted)
#sfl.sigmoidArrayAndAggregatePlot(sigmoidokminfirst)

import sys
sys.exit()

