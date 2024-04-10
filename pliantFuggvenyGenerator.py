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
    for i in range(xmin, xmax):
        xcorr.append(i)

    ycorr = []
    for i in range(xmin, xmax):
        ycorr.append(pliantPoint(alpha, a, i))

    return np.array([xcorr, ycorr])


def pliantFnTombAbrazolas(pliantTomb):
    plt.figure().set_figheight(2.5)
    for pfgv in pliantTomb:
        plt.plot(pfgv[0], pfgv[1])


    plt.grid()
    plt.show()


pliantok = [pliantFn(1, 7, 0, 14)]
for i in range(5):
    alpha = random.randint(-3, 3)
    a = random.randint(7, 40)
    xmin = a - 7
    xmax = a + 7
    pliantok.append(pliantFn(alpha, a, xmin, xmax))


pliantFnTombAbrazolas(pliantok)
