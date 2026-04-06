import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn . linear_model import LogisticRegression


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# A

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
            cmap='bwr', label="train", s=5)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
            cmap='bwr', label="test", s=5, marker="X")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()

# B

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

# C

print("Intercept i koeficijent: ",
      LogRegression_model.intercept_, LogRegression_model.coef_)
theta0 = LogRegression_model.intercept_[0]
theta1 = LogRegression_model.coef_[0][0]
theta2 = LogRegression_model.coef_[0][1]
x1 = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 2)
x2 = (theta0-theta1*x1)/theta2
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="bwr", s=5)
plt.plot(x1, x2, color="black", label="granica odluke")
plt.show()

# D

y_test_p = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
print("Matrica zabune: \n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

accuracy = accuracy_score(y_test, y_test_p)
precision = precision_score(y_test, y_test_p)
recall = recall_score(y_test, y_test_p)

print("Tocnost: ", accuracy)
print("Preciznost: ", precision)
print("Odziv: ", recall)

#E

is_correct = (y_test_p == y_test)
plt.figure()
plt.scatter(X_test[is_correct, 0], X_test[is_correct, 1], color="green", label="dobro")
plt.scatter(X_test[~is_correct, 0], X_test[~is_correct, 1],color="red", label="pogresno")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Dobro vs pogresno klasificirani primjeri")
plt.legend()
plt.show()
