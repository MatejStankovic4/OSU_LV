numbers=[]
while True:
    unos=input()
    if unos=="Done" :
        break
    try:
        number=float(unos)
        numbers.append(number)
    except:
        print("Error")
print(numbers)
numbers.sort()
numberOfNumbers=len(numbers)
print("Min number ",min(numbers))
print("Max number ",max(numbers))
print("Average is:",sum(numbers)/numberOfNumbers)

