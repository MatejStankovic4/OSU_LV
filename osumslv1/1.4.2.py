
try:
    mark=float(input("Unesi broj"))
    if mark<0.0 or mark>1.0:
        print("Broj nije iz intervala")
    elif mark>=0.9:
        print("A")
    elif mark>=0.8:
        print("B")
    elif mark>=0.7:
        print("C")
    elif mark>=0.6:
        print("D")
    else:
        print("F")
except:
    print("Broj nije unesen")