import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.use('TkAgg')


#alphaI = int(input("alpha"))
#aI = int(input("a"))
#xminI = int(input("x min"))
#xmaxI = int(input("x max"))


def pliantPoint(alpha, a, x):
    return 1/(1+math.exp(-alpha*(x-a)))


def pliantFn(alpha, a, xmin, xmax):
    xcorr = []
    for i in range(xmin, xmax+1):
        xcorr.append(i)

    ycorr = []
    for i in range(xmin, xmax+1):
        ycorr.append(pliantPoint(alpha, a, i))

    ycorr = alphaTransform(ycorr, alpha)

    return np.array([xcorr, ycorr])


def alphaTransform(ycorrs, alpha):
    if alpha < 0:
        for i in range(len(ycorrs)):
            ycorrs[i] /= 2
    elif alpha > 0:
        for i in range(len(ycorrs)):
            ycorrs[i] /= 2
            ycorrs[i] += 0.5
    return ycorrs


def pliantFnTombAbrazolas(pliantTomb):
    plt.figure().set_figheight(2.5)
    for pfgv in pliantTomb:
        plt.plot(pfgv[0], pfgv[1])

    plt.grid()
    plt.show()


pliantok = [pliantFn(-1, 7, 0, 14)]
for i in range(5):
    alpha = random.uniform(-3, 3)
    a = random.randint(7, 40)
    xmin = a - 7
    xmax = a + 7
    pliantok.append(pliantFn(alpha, a, xmin, xmax))


pliantFnTombAbrazolas(pliantok)
