import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#Zad 1
unique_colors = np.unique(img_array_aprox, axis=0)
print("Broj boja slika 1: ", len(unique_colors))

#Zad 2
km = KMeans(n_clusters=5, init="random", n_init=5, random_state=0)
km.fit(img_array_aprox)
Y = km.predict(img_array_aprox)

#Zad 3
centroids = km.cluster_centers_
img_array_aprox_ = centroids[Y] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada

#Zad 4
img_aprox = np.reshape(img_array_aprox_, (w, h, d))
plt.figure()
plt.imshow(img_aprox)
plt.title("Kvantizirana slika")

#promjena broja klastera
km50 = KMeans(n_clusters=50, init="random", n_init=5, random_state=0)
km50.fit(img_array_aprox)
Y50 = km50.predict(img_array_aprox)
centroids50 = km50.cluster_centers_
img_array_aprox50 = centroids50[Y50] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada
img_aprox50 = np.reshape(img_array_aprox50, (w, h, d))
plt.figure()
plt.imshow(img_aprox50)
plt.title("Kvantizirana slika - 50 klastera")

#Zad 5

#2. slika
# ucitaj sliku
img2 = Image.imread("imgs\\imgs\\test_2.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika - 2")
plt.imshow(img2)
plt.tight_layout()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img2 = img2.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w2,h2,d2 = img2.shape
img2_array = np.reshape(img2, (w2*h2, d2)) #svaki red je jedan piksel slike, a stupci su RGB komponente

# rezultatna slika
img2_array_aprox = img2_array.copy()

#Zad 1
unique_colors = np.unique(img2_array_aprox, axis=0)
print("Broj boja slika 2: ", len(unique_colors))

#Zad 2
km2 = KMeans(n_clusters=5, init="random", n_init=5, random_state=0)
km2.fit(img2_array_aprox)
Y2 = km2.predict(img2_array_aprox)

#Zad 3
centroids2 = km2.cluster_centers_
img2_array_aprox_ = centroids2[Y2] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada

#Zad 4
img2_aprox = np.reshape(img2_array_aprox_, (w2, h2, d2))
plt.figure()
plt.imshow(img2_aprox)
plt.title("Kvantizirana slika - 2")

#promjena broja klastera
km50_2 = KMeans(n_clusters=50, init="random", n_init=5, random_state=0)
km50_2.fit(img2_array_aprox)
Y50_2 = km50_2.predict(img2_array_aprox)
centroids50_2 = km50_2.cluster_centers_
img2_array_aprox50 = centroids50_2[Y50_2] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada
img2_aprox50 = np.reshape(img2_array_aprox50, (w2, h2, d2))
plt.figure()
plt.imshow(img2_aprox50)
plt.title("Kvantizirana slika - 2 - 50 klastera")




#3. slika
# ucitaj sliku
img3 = Image.imread("imgs\\imgs\\test_3.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika - 3")
plt.imshow(img3)
plt.tight_layout()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img3 = img3.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w3,h3,d3 = img3.shape
img3_array = np.reshape(img3, (w3*h3, d3)) #svaki red je jedan piksel slike, a stupci su RGB komponente

# rezultatna slika
img3_array_aprox = img3_array.copy()

#Zad 1
unique_colors = np.unique(img3_array_aprox, axis=0)
print("Broj boja slika 3: ", len(unique_colors))

#Zad 2
km3 = KMeans(n_clusters=5, init="random", n_init=5, random_state=0)
km3.fit(img3_array_aprox)
Y3 = km3.predict(img3_array_aprox)

#Zad 3
centroids3 = km3.cluster_centers_
img3_array_aprox_ = centroids3[Y3] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada

#Zad 4
img3_aprox = np.reshape(img3_array_aprox_, (w3, h3, d3))
plt.figure()
plt.imshow(img3_aprox)
plt.title("Kvantizirana slika - 3")

#promjena broja klastera
km50_3 = KMeans(n_clusters=50, init="random", n_init=5, random_state=0)
km50_3.fit(img3_array_aprox)
Y50_3 = km50_3.predict(img3_array_aprox)
centroids50_3 = km50_3.cluster_centers_
img3_array_aprox50 = centroids50_3[Y50_3] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada
img3_aprox50 = np.reshape(img3_array_aprox50, (w3, h3, d3))
plt.figure()
plt.imshow(img3_aprox50)
plt.title("Kvantizirana slika - 3 - 50 klastera")



#4. slika
# ucitaj sliku
img4 = Image.imread("imgs\\imgs\\test_4.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika - 4")
plt.imshow(img4)
plt.tight_layout()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img4 = np.clip(img4.astype(np.float64), 0, 1)

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w4,h4,d4 = img4.shape
img4_array = np.reshape(img4, (w4*h4, d4)) #svaki red je jedan piksel slike, a stupci su RGB komponente

# rezultatna slika
img4_array_aprox = img4_array.copy()

#Zad 1
unique_colors = np.unique(img4_array_aprox, axis=0)
print("Broj boja slika 4: ", len(unique_colors))

#Zad 2
km4 = KMeans(n_clusters=5, init="random", n_init=5, random_state=0)
km4.fit(img4_array_aprox)
Y4 = km4.predict(img4_array_aprox)

#Zad 3
centroids4 = km4.cluster_centers_
img4_array_aprox_ = centroids4[Y4] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada

#Zad 4
img4_aprox = np.reshape(img4_array_aprox_, (w4, h4, d4))
plt.figure()
plt.imshow(img4_aprox)
plt.title("Kvantizirana slika - 4")

#promjena broja klastera
km50_4 = KMeans(n_clusters=50, init="random", n_init=5, random_state=0)
km50_4.fit(img4_array_aprox)
Y50_4 = km50_4.predict(img4_array_aprox)
centroids50_4 = km50_4.cluster_centers_
img4_array_aprox50 = centroids50_4[Y50_4] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada
img4_aprox50 = np.reshape(img4_array_aprox50, (w4, h4, d4))
plt.figure()
plt.imshow(img4_aprox50)
plt.title("Kvantizirana slika - 4 - 50 klastera")



#5. slika
# ucitaj sliku
img5 = Image.imread("imgs\\imgs\\test_5.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika - 5")
plt.imshow(img5)
plt.tight_layout()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img5 = np.clip(img5.astype(np.float64) / 255, 0, 1)

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w5,h5,d5 = img5.shape
img5_array = np.reshape(img5, (w5*h5, d5)) #svaki red je jedan piksel slike, a stupci su RGB komponente

# rezultatna slika
img5_array_aprox = img5_array.copy()

#Zad 1
unique_colors = np.unique(img5_array_aprox, axis=0)
print("Broj boja slika 5: ", len(unique_colors))

#Zad 2
km5 = KMeans(n_clusters=5, init="random", n_init=5, random_state=0)
km5.fit(img5_array_aprox)
Y5 = km5.predict(img5_array_aprox)

#Zad 3
centroids5 = km5.cluster_centers_
img5_array_aprox_ = centroids5[Y5] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada

#Zad 4
img5_aprox = np.reshape(img5_array_aprox_, (w5, h5, d5))
plt.figure()
plt.imshow(img5_aprox)
plt.title("Kvantizirana slika - 5")

#promjena broja klastera
km50_5 = KMeans(n_clusters=50, init="random", n_init=5, random_state=0)
km50_5.fit(img5_array_aprox)
Y50_5 = km50_5.predict(img5_array_aprox)
centroids50_5 = km50_5.cluster_centers_
img5_array_aprox50 = centroids50_5[Y50_5] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada
img5_aprox50 = np.reshape(img5_array_aprox50, (w5, h5, d5))
plt.figure()
plt.imshow(img5_aprox50)
plt.title("Kvantizirana slika - 5 - 50 klastera")



#6. slika
# ucitaj sliku
img6 = Image.imread("imgs\\imgs\\test_6.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika - 6")
plt.imshow(img6)
plt.tight_layout()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img6 = np.clip(img6.astype(np.float64) / 255, 0, 1)

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w6,h6,d6 = img6.shape
img6_array = np.reshape(img6, (w6*h6, d6)) #svaki red je jedan piksel slike, a stupci su RGB komponente

# rezultatna slika
img6_array_aprox = img6_array.copy()

#1
unique_colors = np.unique(img6_array_aprox, axis=0)
print("Broj boja slika 6: ", len(unique_colors))

#2
km6 = KMeans(n_clusters=5, init="random", n_init=5, random_state=0)
km6.fit(img6_array_aprox)
Y6 = km6.predict(img6_array_aprox)

#3
centroids6 = km6.cluster_centers_
img6_array_aprox_ = centroids6[Y6] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada

#4
img6_aprox = np.reshape(img6_array_aprox_, (w6, h6, d6))
plt.figure()
plt.imshow(img6_aprox)
plt.title("Kvantizirana slika - 6")

#promjena broja klastera
km50_6 = KMeans(n_clusters=50, init="random", n_init=5, random_state=0)
km50_6.fit(img6_array_aprox)
Y50_6 = km50_6.predict(img6_array_aprox)
centroids50_6 = km50_6.cluster_centers_
img6_array_aprox50 = centroids50_6[Y50_6] #svaki piksel slike zamijenjen je centroidom odn. boji kojoj pripada
img6_aprox50 = np.reshape(img6_array_aprox50, (w6, h6, d6))
plt.figure()
plt.imshow(img6_aprox50)
plt.title("Kvantizirana slika - 6 - 50 klastera")

#6
sample_idx = np.random.choice(img_array_aprox.shape[0], size=int(0.1*img_array_aprox.shape[0]), replace=False)
img_sample = img_array_aprox[sample_idx]

K_values = range(1, 21) 
inertia_values = []

for K in K_values:
    km = KMeans(n_clusters=K, init="random", n_init=5, random_state=0)
    km.fit(img_sample)
    inertia_values.append(km.inertia_)

plt.figure()
plt.plot(K_values, inertia_values, marker='o')
plt.title("Slika 1 - lakat metoda")
plt.xlabel("Broj klastera K")
plt.ylabel("J (inertia)")

sample_idx2 = np.random.choice(img2_array_aprox.shape[0], size=int(0.1*img2_array_aprox.shape[0]), replace=False)
img_sample2 = img2_array_aprox[sample_idx2]

K_values = range(1, 21) 
inertia_values = []

for K in K_values:
    km = KMeans(n_clusters=K, init="random", n_init=5, random_state=0)
    km.fit(img_sample2)
    inertia_values.append(km.inertia_)

plt.figure()
plt.plot(K_values, inertia_values, marker='o')
plt.title("Slika 2 - lakat metoda")
plt.xlabel("Broj klastera K")
plt.ylabel("J (inertia)")

sample_idx3 = np.random.choice(img3_array_aprox.shape[0], size=int(0.1*img3_array_aprox.shape[0]), replace=False)
img_sample3 = img3_array_aprox[sample_idx3]

K_values = range(1, 21) 
inertia_values = []

for K in K_values:
    km = KMeans(n_clusters=K, init="random", n_init=5, random_state=0)
    km.fit(img_sample3)
    inertia_values.append(km.inertia_)

plt.figure()
plt.plot(K_values, inertia_values, marker='o')
plt.title("Slika 3 - lakat metoda")
plt.xlabel("Broj klastera K")
plt.ylabel("J (inertia)")

sample_idx4 = np.random.choice(img4_array_aprox.shape[0], size=int(0.1*img4_array_aprox.shape[0]), replace=False)
img_sample4 = img4_array_aprox[sample_idx4]

K_values = range(1, 21) 
inertia_values = []

for K in K_values:
    km = KMeans(n_clusters=K, init="random", n_init=5, random_state=0)
    km.fit(img_sample4)
    inertia_values.append(km.inertia_)

plt.figure()
plt.plot(K_values, inertia_values, marker='o')
plt.title("Slika 4 - lakat metoda")
plt.xlabel("Broj klastera K")
plt.ylabel("J (inertia)")

sample_idx5 = np.random.choice(img5_array_aprox.shape[0], size=int(0.1*img5_array_aprox.shape[0]), replace=False)
img_sample5 = img5_array_aprox[sample_idx5]

K_values = range(1, 21) 
inertia_values = []

for K in K_values:
    km = KMeans(n_clusters=K, init="random", n_init=5, random_state=0)
    km.fit(img_sample5)
    inertia_values.append(km.inertia_)

plt.figure()
plt.plot(K_values, inertia_values, marker='o')
plt.title("Slika 5 - lakat metoda")
plt.xlabel("Broj klastera K")
plt.ylabel("J (inertia)")

sample_idx6 = np.random.choice(img6_array_aprox.shape[0], size=int(0.1*img6_array_aprox.shape[0]), replace=False)
img_sample6 = img6_array_aprox[sample_idx6]

K_values = range(1, 21) 
inertia_values = []

for K in K_values:
    km = KMeans(n_clusters=K, init="random", n_init=5, random_state=0)
    km.fit(img_sample6)
    inertia_values.append(km.inertia_)

plt.figure()
plt.plot(K_values, inertia_values, marker='o')
plt.title("Slika 6 - lakat metoda")
plt.xlabel("Broj klastera K")
plt.ylabel("J (inertia)")

#7
images = [img, img2, img3, img4, img5, img6]       
Y_list = [Y, Y2, Y3, Y4, Y5, Y6 ]                 
sizes = [(w,h), (w2,h2), (w3,h3), (w4,h4), (w5,h5), (w6,h6)]
num_clusters = 5
for img_idx, Y in enumerate(Y_list):
    w, h = sizes[img_idx]
    plt.figure()
    plt.suptitle(f"Maske klastera za sliku {img_idx+1}")
    for i in range(num_clusters):
        mask = (Y == i).astype(np.float64)
        mask_image = np.reshape(mask, (w, h))
        plt.subplot(1, num_clusters, i+1)
        plt.imshow(mask_image, cmap='gray')
        plt.title(f"Klaster {i+1}")
plt.show()
