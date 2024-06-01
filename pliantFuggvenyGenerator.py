import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.use('TkAgg')


def pliantPoint(alpha, a, x):
    return 1/(1 + np.exp(-alpha*(x - a)))


def pliantFn(alpha, a, xmin, xmax):
    xcorr = np.arange(xmin, xmax + 1)
    ycorr = pliantPoint(alpha, a, xcorr)

    ycorr = alphaTransform(ycorr, alpha)

    return np.array([xcorr, ycorr])


def alphaTransform(ycorrs, alpha):
    if alpha < 0:
        ycorrs /= 2
    elif alpha > 0:
        ycorrs = ycorrs / 2 + 0.5
    return ycorrs


def pliantSigmoidAggregate(pliantTomb):
    return 1


def pliantArrayShow(pliantTomb):
    plt.figure().set_figheight(2.5)
    for pfgv in pliantTomb:
        plt.plot(pfgv[0], pfgv[1])

    plt.grid()
    plt.show()


def main():
    plotmin = 0
    plotmax = 50

    pliantok = [pliantFn(-1, 7, plotmin, plotmax)]
    for i in range(5):
        alpha = random.uniform(-3, 3)
        a = random.randint(7, 40)
        pliantok.append(pliantFn(alpha, a, plotmin, plotmax))

    pliantArrayShow(pliantok)


if __name__ == "__main__":
    main()
