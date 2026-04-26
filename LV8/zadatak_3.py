import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image

model = load_model("FCN.keras")
model.summary()

img = Image.open("test.png").convert("L")
img = img.resize((28, 28))
img_array = np.array(img)
img_array = img_array.astype("float32") / 255

img_array = np.expand_dims(img_array, 0)
img_array = np.expand_dims(img_array, -1)

img_pred= model.predict(img_array)
pred_class = np.argmax(img_pred)

print(f"Predvideno: {pred_class}")
plt.imshow(img_array[0, :, :, 0], cmap="gray")
plt.title(f"Predvideno: {pred_class}")
plt.show()
