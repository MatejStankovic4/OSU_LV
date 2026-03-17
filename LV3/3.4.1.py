import pandas as pd
import numpy as np
import matplotlib . pyplot as plt



data = pd.read_csv('data_C02_emission.csv')

#A

#print(len(data))
#print(data.info())
#data.drop_duplicates()
#data=data.reset_index(drop=True)

#B

#sorted = data.sort_values(by='Fuel Consumption City (L/100km)')
#print(sorted)
#least_consuming = sorted.tail(3)
#print("Least consuming: ")
#print(least_consuming[['Make', 'Vehicle Class', 'Fuel Consumption City (L/100km)']])
#most_consuming = sorted.head(3)
#print("Most consuming: ")
#print(most_consuming[['Make', 'Vehicle Class', 'Fuel Consumption City (L/100km)']])

#C

#engine_size = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
##print(engine_size.iloc[:,3:4])
#print("Number of cars with <2.5,3.5> engine: ",len(engine_size))
#print("Mean value of CO2 Emissions: ",engine_size['CO2 Emissions (g/km)'].mean())

#D

#audi_cars = data[(data['Make'] == "Audi")]
#print("Number of Audi cars: ",len(audi_cars))
#audiWith4Cylinders = audi_cars[(audi_cars['Cylinders'] == 4)]
#print(audiWith4Cylinders)
#print("Mean value of CO2 Emissions for 4cylinders Audi: ", audiWith4Cylinders['CO2 Emissions (g/km)'].mean())

#E

#With4Cylinders = data[(data['Cylinders'] == 4)]
#print(len(With4Cylinders))
#print(With4Cylinders['CO2 Emissions (g/km)'].mean())
#With6Cylinders = data[(data['Cylinders'] == 6)]
#print(len(With6Cylinders))
#print(With6Cylinders['CO2 Emissions (g/km)'].mean())
#With8Cylinders = data[(data['Cylinders'] == 8)]
#print(len(With8Cylinders))
#print(With8Cylinders['CO2 Emissions (g/km)'].mean())
#With10Cylinders = data[(data['Cylinders'] == 10)]
#print(len(With10Cylinders))
#print(With10Cylinders['CO2 Emissions (g/km)'].mean())
#With12Cylinders = data[(data['Cylinders'] == 12)]
#print(len(With12Cylinders))
#print(With12Cylinders['CO2 Emissions (g/km)'].mean())
#With16Cylinders = data[(data['Cylinders'] == 16)]
#print(len(With16Cylinders))
#print(With16Cylinders['CO2 Emissions (g/km)'].mean())

#F

#GasolineFuel = data[(data['Fuel Type'] == 'X') | (data['Fuel Type'] == 'Z')]
#print(GasolineFuel['Fuel Consumption City (L/100km)'].mean())
#DieselFuel = data[data['Fuel Type']=='D']
#print(DieselFuel['Fuel Consumption City (L/100km)'].mean())

#G

#Diesel4Cylinders = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
#Diesel4Cylinders.sort_values(by = 'Fuel Consumption City (L/100km)')
#mostFuelConsumption = Diesel4Cylinders.head(1)
#print(mostFuelConsumption)

#H

#manualGears = data[data['Transmission'].str.startswith(('AM', 'M'), na=False)]
#print("Cars with manual gears: ", len(manualGears))

#I

#print(data.corr(numeric_only=True))


