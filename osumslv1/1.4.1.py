def total_euro(workHours, payment):
    return float(workHours)*float(payment)
hours=float(input("Radni sati: "))
payment=float(input("Iznos po satu: "))
total= total_euro(hours,payment)
print("Ukupna zarada: ", total)