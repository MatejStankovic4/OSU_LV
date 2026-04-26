import numpy as np
from tensorflow import keras
from keras.models import load_model
from matplotlib import pyplot as plt

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

model = load_model("FCN.keras")
model.summary()

y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1) #za svaki primjer odredi koja je najveca vjerojatnost da pripada nekoj klasi
count = 0
for i in range(len(y_pred_classes)):
    if count == 5:
        break
    if y_pred_classes[i] != y_test[i]:
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f"Stvarna: {y_test[i]}, Predikcija: {y_pred_classes[i]}")
        plt.axis('off')
        plt.show()
        count += 1
