import re
import pandas as pd

def percent2float(string):
    return float(re.sub("[^0-9]+", "", string)) / 100

def load_data(filename="dset.csv"):
    conv = {"damage taken": percent2float}
    data = pd.read_csv("dset.csv", converters=conv)
    data[['month', 'year']] = data.pop('month of departure').str.split(
        pat='/',
        n=1,
        expand=True,
    )
    for k in ["month", "year"]:
        data[k] = data[k].astype(int)
    data["tod"] = 12 * (data["year"] - 1396) + data["month"]
    return data

