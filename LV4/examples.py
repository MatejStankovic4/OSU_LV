from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()

import sklearn.linear_model as lm
linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error
y_test_p = linearModel.predict(X_test_n)
MAE = mean_absolute_error(y_test, y_test_p)

