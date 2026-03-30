import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np


data = pd.read_csv('data_C02_emission.csv')

#A

numerical_features = ["Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)"]
X = data[numerical_features]
y = data["CO2 Emissions (g/km)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

#B

train_df = X_train.copy()
train_df["CO2 Emissions (g/km)"] = y_train
train_df["type"] = "train"

test_df = X_test.copy()
test_df["CO2 Emissions (g/km)"] = y_test
test_df["type"] = "test"

plot_df = pd.concat([train_df, test_df]).reset_index(drop=True)
colors = {"train": "blue", "test": "red"}
plot_df.plot.scatter(x = "Fuel Consumption City (L/100km)", y = "CO2 Emissions (g/km)", color = plot_df["type"].map(colors), s=5)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")

plot_df.plot.scatter(x = "Fuel Consumption Hwy (L/100km)", y = "CO2 Emissions (g/km)", color = plot_df["type"].map(colors), s=5)
plt.xlabel("Fuel Consumption Hwy (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")

plot_df.plot.scatter(x = "Fuel Consumption Comb (L/100km)", y = "CO2 Emissions (g/km)", color = plot_df["type"].map(colors), s=5)
plt.xlabel("Fuel Consumption Comb (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")

#C

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_train_n_df = pd.DataFrame(X_train_n, columns=X_train.columns)
plt.figure()
X_train["Fuel Consumption City (L/100km)"].plot(kind="hist", bins=20)
plt.figure()
X_train_n_df["Fuel Consumption City (L/100km)"].plot(kind="hist", bins=20)

X_test_n = sc.transform(X_test)
X_test_n_df = pd.DataFrame(X_test_n, columns=X_test.columns)

#D

linearModel1 = lm.LinearRegression()
linearModel1.fit(X_train_n_df[["Fuel Consumption City (L/100km)"]], y_train)
print("Za potrosnju u gardu:")
print("Intercept:", linearModel1.intercept_) #vrijednost kada su sve pojedine ulazne veličine jednake prosječnoj vrijednosti, odnosno nuli
print("Koeficijenti: ", linearModel1.coef_) #koliko koja ulazna veličina utječe na izlaz

linearModel2 = lm.LinearRegression()
linearModel2.fit(X_train_n_df[["Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)"]], y_train)
print("Za potrosnju na autocesti:")
print("Intercept:", linearModel2.intercept_) #vrijednost kada su sve pojedine ulazne veličine jednake prosječnoj vrijednosti, odnosno nuli
print("Koeficijenti: ", linearModel2.coef_) #koliko koja ulazna veličina utječe na izlaz

linearModel3 = lm.LinearRegression()
linearModel3.fit(X_train_n_df[["Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)"]], y_train)
print("Za potrosnju u kombinaciji:")
print("Intercept:", linearModel3.intercept_) #vrijednost kada su sve pojedine ulazne veličine jednake prosječnoj vrijednosti, odnosno nuli
print("Koeficijenti: ", linearModel3.coef_) #koliko koja ulazna veličina utječe na izlaz

#E

y_test_p1 = linearModel1.predict(X_test_n_df[["Fuel Consumption City (L/100km)"]])
y_test_p2 = linearModel2.predict(X_test_n_df[["Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)"]])
y_test_p3 = linearModel3.predict(X_test_n_df[["Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)"]])
plt.figure()
plt.scatter(x = y_test, y = y_test_p1, s=5)
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predviđene vrijednosti")
plt.figure()
plt.scatter(x = y_test, y = y_test_p2, s=5)
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predviđene vrijednosti")
plt.figure()
plt.scatter(x = y_test, y = y_test_p3, s=5)
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predviđene vrijednosti")
plt.show()

#F

print("Za potrosnju u gradu - greske:")
mse = mean_squared_error(y_test, y_test_p1)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_p1)
mape = mean_absolute_percentage_error(y_test, y_test_p1)
r2 = r2_score(y_test, y_test_p1)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R^2: {r2:.4f}")

print("Za potrosnju na autocesti i gradu - greske:")
mse = mean_squared_error(y_test, y_test_p2)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_p2)
mape = mean_absolute_percentage_error(y_test, y_test_p2)
r2 = r2_score(y_test, y_test_p2)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R^2: {r2:.4f}")

print("Za potrosnju u kombinaciji svi - greske:")
mse = mean_squared_error(y_test, y_test_p3)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_p3)
mape = mean_absolute_percentage_error(y_test, y_test_p3)
r2 = r2_score(y_test, y_test_p3)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R^2: {r2:.4f}")
