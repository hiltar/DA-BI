import csv
import matplotlib.pyplot as plt
import sklearn

with open('data/vauvanimet.csv') as f:
    reader = csv.reader(f) ## Avaa readerin
    for row in reader:
        print(row)
## 