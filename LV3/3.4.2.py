import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('e:\softeng\osu_lv\lv3\data_C02_emission.csv')

#A

plt.figure ()
data ['CO2 Emissions (g/km)'].plot(kind='hist' , bins = 20 )
plt.show ()

#B

colors = data['Fuel Type'].map({
    'X': 'red',
    'Z': 'blue',
    'D': 'green',
    'E': 'orange',
    'N': 'yellow'
})

plt.figure ()
plt.scatter(data['Fuel Consumption City (L/100km)'],data['CO2 Emissions (g/km)'], s=5, c=colors)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Odnos izme ¯du gradske potrošnje goriva i emisije C02 plinova ")
plt.show ()
print("S povečanjem potrošnje goriva povećava se emisija CO2 u gradovima")

#C

plt.figure ()
data.boxplot(column= 'Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.xlabel("Tip goriva")
plt.ylabel("Fuel Consumption Hwy (L/100km)")
plt.show ()

print("nema mjerenja za prirodna goriva i kod premium benzina dolazi do velikih odstupanja ")

#D

fuel= data.groupby('Fuel Type').size()

plt.figure()
plt.bar(fuel.index, fuel.values)
plt.title("Broj vozila po tipu goriva")
plt.xlabel("Tip goriva")
plt.ylabel("Broj vozila")
plt.show()

#E

cyl= data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().sort_index()

plt.figure()
plt.bar(cyl.index, cyl.values)
plt.xticks(cyl.index)
plt.xlabel("Broj cilindara")
plt.ylabel("Prosječna CO2 emisija (g/km)")
plt.title("Prosječna CO2 emisija vozila prema broju cilindara")

plt.show()
