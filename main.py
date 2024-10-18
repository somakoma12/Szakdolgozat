import sigmoidFunctionLib as sfl

import numpy as np
import random

plotmin = -100
plotmax = 400
#params setup
params = sfl.createParamsArray(1/4, 100, plotmin, plotmax, True)
params = sfl.appendParams(params, 1/4, 140, plotmin, plotmax, False)
params = sfl.appendParams(params, 1/4, 200, plotmin, plotmax, False)
params = sfl.appendParams(params, 1/4, 250, plotmin, plotmax, True)
#slice first sigmoid
paramsminfirst = params[1:4]

#params into sigmoids
sigmoidok = sfl.sigmoidArray(params)
sigmoidokminfirst = sfl.sigmoidArray(paramsminfirst)

#sigmoids aggregated
aggregate = sfl.sigmoidAggregate(sigmoidok)
aggregateminfirst = sfl.sigmoidAggregate(sigmoidokminfirst)

#subtracting first sigmoid
kivonas = sfl.subtractFirstSigmoidByDivideKnowingAlpha(aggregate, 1/4)

#plots
#sfl.aggregateShow(aggregate)
#sfl.aggregateShow(aggregateminfirst)
#sfl.aggregateShow(kivonas)
#sfl.sigmoidArrayAndAggregatePlot(sigmoidokminfirst)

import sys
sys.exit()

