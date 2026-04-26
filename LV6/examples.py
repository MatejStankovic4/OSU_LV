import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_n , y_train)

SVM_model = svm.SVC(kernel='rbf', gamma = 1, C=0.1)
SVM_model.fit(X_train_n , y_train)

y_test_p_KNN = KNN_model.predict(X_test)
y_test_p_SVM = SVM_model.predict(X_test)