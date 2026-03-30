import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('data_C02_emission.csv')

#A

numerical_features = ["Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)"]
X = data[numerical_features]

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[["Fuel Type"]]).toarray()
X_encoded = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(["Fuel Type"]))   

X = pd.concat([X, X_encoded], axis = 1)
y = data["CO2 Emissions (g/km)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

#D

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)
print("Intercept:", linearModel.intercept_) #vrijednost kada su sve pojedine ulazne veličine jednake prosječnoj vrijednosti, odnosno nuli
print("Koeficijenti: ", linearModel.coef_)

#E

y_test_p = linearModel.predict(X_test)
plt.figure()
plt.scatter(x = y_test, y = y_test_p, s=5)
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predviđene vrijednosti")
plt.show()

#F

mse = mean_squared_error(y_test, y_test_p)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_p)
mape = mean_absolute_percentage_error(y_test, y_test_p)
r2 = r2_score(y_test, y_test_p)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R^2: {r2:.4f}")

#maksimalna pogreška

errors = np.abs(y_test - y_test_p)
print("Maksimalna pogreska (g/km):", errors.max())
max_error_index= errors.idxmax()
worst_vehicle = data.loc[max_error_index]
print("Vozilo:", worst_vehicle)
