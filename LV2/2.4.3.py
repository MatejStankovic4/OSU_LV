import numpy as np
import matplotlib . pyplot as plt
import matplotlib.image as img
from matplotlib import transforms

image=img.imread("C:/Users/student/Desktop/LV2 Rad s bibliotekama Numpy i Matplotlib-20260311/road.jpg")
image=image.copy()
plt.figure()
plt.imshow(image,alpha=0.6)
plt.show()

širina = len(image[0])
četvrtina = int(širina/4)
plt.imshow(image[:, 1*četvrtina: 2*četvrtina , :])
plt.show() 


rotirana=np.rot90(image,3)
plt.imshow(rotirana)
plt.show()

zrcalo= np.fliplr(image)
plt.imshow(zrcalo)
plt.show()