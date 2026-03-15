import numpy as np
import matplotlib . pyplot as plt
data=np.loadtxt("C:/Users/student/Desktop/LV2 Rad s bibliotekama Numpy i Matplotlib-20260311/data.csv",dtype="str",skiprows=1, delimiter=",")
data = np.array(data,np.float64)
print("Broj osoba:",len(data))
height=(data[:,1])
weight=(data[:,2])
plt.scatter(height,weight,s=5)
plt.xlabel("height")
plt.ylabel("weight")
plt.show()
height50=(height[::50])
weight50=(weight[::50])
plt.scatter(height50,weight50)
plt.xlabel("height")
plt.ylabel("weight")
plt.show()
print("Min. height: ", height.min())
print("Max. height: ", height.max())
print("Mean: ", height.mean())

indM = (data[:,0] == 1)
indW = (data[:,0]==0)
men=data[indM]
women=data[indW]

men_height=(data[:,1])
women_height=(data[:,1])

print("Min. man height: ", men_height.min())
print("Max. man height: ", men_height.max())
print("Mean man height: ", men_height.mean())

print("\nMin. woman height: ", women_height.min())
print("Max. woman height: ", women_height.max())
print("Mean woman height: ", women_height.mean())