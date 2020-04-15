import csv

with open('data/vauvanimet.csv') as f:
    reader = csv.reader(f) ## Avaa readerin
    for row in reader:
        print(row)
        