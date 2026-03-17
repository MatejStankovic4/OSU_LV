import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s1 = pd.Series(['crvenkapica', 'baka', 'majka', 'otac', 'vuk'])
#print(s1)
s2 = pd.Series(5., index=['a', 'b', 'c', 'd', 'e'], name='ime_objekta')
#print(s2)
#print(s2['b'])

s3 = pd.Series(np.random.randn(5))
#print(s3)
#print(s3[3])

data = {'country' : ['Italy', 'Spain', 'Greece', 'Portugal', 'France'],'population': [60.6, 46.7, 10.7, 10.3, 67.0], 'code' : [39,34,30,33,351]}
countries = pd.DataFrame(data, columns=['country','population','code'])
#print(countries)



data = pd.read_csv('data_C02_emission.csv')

#print(len(data))
#print(data)
#print(data.head(5))
#print(data.info())
#print(data.info())
#print(data.max())

#print(data['Cylinders'])
#print(data.Cylinders)

#print(data[['Model', 'Cylinders']])

#print(data.iloc[2:6, 2:7])
#print(data.iloc[:,2:5])
#print(data.iloc[:,[0,4,7]])
#print(data.Cylinders>6)

# print(data[data.Cylinders>6])
#data['jedinice'] = np.ones(len(data))

#new_data = data.groupby('Cylinders')
#print(new_data.count())
#print(new_data.size())

#plt.figure()
#data['Fuel Consumption City (L/100km)'].plot(kind='hist',bins=20)
#plt.figure()
#data['Fuel Consumption City (L/100km)'].plot(kind='box')
#plt.show()

#grouped = data.groupby('Cylinders')
#grouped.boxplot(column=['CO2 Emissions (g/km)']) 
#data.boxplot(column=['CO2 Emissions (g/km)'], by='Cylinders')
#plt.show()

#data.plot.scatter(x='Fuel Consumption City (L/100km)' , y='CO2 Emissions (g/km)', c='Engine Size (L)', cmap='hot', s=50)
#plt.show()

#print(data.corr(numeric_only=True))


            