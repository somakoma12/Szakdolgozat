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
    print(1/(1+math.exp(-alpha*(x+a))))
    return 1/(1+math.exp(-alpha*(x+a)))
def pliantFn(alpha, a, xmin, xmax):
    xcorr = []
    for i in range(xmin, xmax):
        xcorr.append(i)

    ycorr = []
    for i in range(xmin, xmax):
        ycorr.append(pliantPoint(alpha, a, i))

    print(xcorr)
    print(xcorr[0])
    print(ycorr)
    print(ycorr[0])

    plt.plot(xcorr, ycorr, color='red')
    plt.grid()
    plt.show()


#pliantFn(alphaI, aI, xmennyI)
pliantFn(1, 0, -10, 10)

