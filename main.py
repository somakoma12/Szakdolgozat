import sigmoidFunctionLib as sfl
import numpy as np
import random

plotmin = -150
plotmax = 150


params = sfl.createParamsArray(1/4, 0, plotmin, plotmax)
params = sfl.appendParams(params, -1/4, 10, plotmin, plotmax)

params = sfl.appendParams(params, -1/4, 30, plotmin, plotmax)
params = sfl.appendParams(params, 1/4, 40, plotmin, plotmax)

for i in range(0):
    params = np.append(params, [[random.uniform(-1/8, 1/8), random.randint(-60, 60), plotmin, plotmax]], axis=0)


sigmoidok = sfl.sigmoidArray(params)
sfl.sigmoidArrayAndAggregatePlot(sigmoidok)

import sys
sys.exit()

