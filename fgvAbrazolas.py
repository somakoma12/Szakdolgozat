import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')

xcorr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ycorr = []

for i in range(10):
    ycorr.append(np.random.randint(20))

print(ycorr)
print(xcorr)
print(xcorr[0])
print(ycorr[0])

plt.plot(xcorr, ycorr, color='red')
plt.grid()
plt.show()
