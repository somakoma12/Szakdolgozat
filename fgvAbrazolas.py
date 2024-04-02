import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')

xcorr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ycorr = np.empty(1, dtype=int)

for i in range(9):
    ycorr = np.insert(ycorr, 0, np.random.randint(20))

print(ycorr)
print(xcorr)
print(xcorr[0])
print(ycorr[0])

plt.plot(xcorr, ycorr, color='red')
plt.grid()
plt.show()
