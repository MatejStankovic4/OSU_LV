import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

km3 = KMeans(n_clusters=3, random_state=0)
km3.fit(X)
Y3 = km3.predict(X)
plt.figure()
plt.scatter(X[:,0],X[:,1], c=Y3)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 1 - 3 grupe')

km6 = KMeans(n_clusters=6, random_state=0)
km6.fit(X)
Y6 = km6.predict(X)
plt.figure()
plt.scatter(X[:,0],X[:,1], c=Y6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 1 - 6 grupa')

X2 = generate_data(500, 2)

plt.figure()
plt.scatter(X2[:,0],X2[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri - 2')

km3.fit(X2)
Y3_2 = km3.predict(X2)
plt.figure()
plt.scatter(X2[:,0],X2[:,1], c=Y3_2)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 2 - 3 grupe')

km6.fit(X2)
Y6_2 = km6.predict(X2)
plt.figure()
plt.scatter(X2[:,0],X2[:,1], c=Y6_2)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 2 - 6 grupa')


X3 = generate_data(500, 3)

plt.figure()
plt.scatter(X3[:,0],X3[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri - 3')

km3.fit(X3)
Y3_3 = km3.predict(X3)
plt.figure()
plt.scatter(X3[:,0],X3[:,1], c=Y3_3)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 3 - 3 grupe')

km4 = KMeans(n_clusters=4, random_state=0)
km4.fit(X3)
Y4_3 = km4.predict(X3)
plt.figure()
plt.scatter(X3[:,0],X3[:,1], c=Y4_3)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 3 - 4 grupe')

km6.fit(X3)
Y6_3 = km6.predict(X3)
plt.figure()
plt.scatter(X3[:,0],X3[:,1], c=Y6_3)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 3 - 6 grupa')


X4 = generate_data(500, 4)

plt.figure()
plt.scatter(X4[:,0],X4[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri - 4')

km2 = KMeans(n_clusters=2, random_state=0)
km2.fit(X4)
Y2_4 = km2.predict(X4)
plt.figure()
plt.scatter(X4[:,0],X4[:,1], c=Y2_4)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 4 - 2 grupe')

Y4_4 = km4.predict(X4)
plt.figure()
plt.scatter(X4[:,0],X4[:,1], c=Y4_4)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 4 - 4 grupe')


X5 = generate_data(500, 5)

plt.figure()
plt.scatter(X5[:,0],X5[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri - 5')

km2.fit(X5)
Y2_5 = km2.predict(X5)
plt.figure()
plt.scatter(X5[:,0],X5[:,1], c=Y2_5)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 5 - 2 grupe')

km4.fit(X5)
Y4_5 = km4.predict(X5)
plt.figure()
plt.scatter(X5[:,0],X5[:,1], c=Y4_5)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri za generiranje 5 - 4 grupe')
plt.show()
