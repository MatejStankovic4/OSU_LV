import numpy as np
import matplotlib . pyplot as plt
zeros= np.zeros((50,50))
ones= np.ones((50,50))
vrh= np.hstack((zeros,ones,zeros))
dno= np.hstack((ones,zeros,ones))

slika = np.vstack((vrh,dno))

plt.figure()
plt.imshow(slika, cmap="gray")
plt.show()
