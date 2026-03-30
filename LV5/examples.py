from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


#LogRegression_model = LogisticRegression()
#LogRegression_model.fit(X_train, y_train)
#y_test_pred = LogRegression_model.predict(X_test)

y_true = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 1, 0, 1, 0, 0])

print("Accuracy:", accuracy_score(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("Matrica zabune: ",cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
disp.plot()
plt.show()

print(classification_report(y_true, y_pred))

plt.figure()
plt.scatter()