import sigmoidFunctionLib as sfl
import numpy as np
import random

plotmin = -100
plotmax = 100

params = sfl.createParamsArray(1/4, -60, plotmin, plotmax, True)
params = sfl.appendParams(params, 1/4, 0, plotmin, plotmax, False)
#params = sfl.appendParams(params, 1/1, -45, plotmin, plotmax, True)
#params = sfl.appendParams(params, 1/4, -40, plotmin, plotmax, False)
#params = sfl.appendParams(params, 1/4, 0, plotmin, plotmax, True)
#params = sfl.appendParams(params, 1/4, 5, plotmin, plotmax, False)





#for i in range(-9, 40, 1):
 #   params = sfl.appendParams(params, 1/4, i, plotmin, plotmax, True)

sigmoidok = sfl.sigmoidArray(params)

aggregate = sfl.sigmoidAggregate(sigmoidok)

sfl.sigmoidArrayAndAggregatePlot(sigmoidok)

import sys
sys.exit()

