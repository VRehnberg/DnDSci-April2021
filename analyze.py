## Imports
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import display
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_context("talk")

## Utilities
def percent2float(string):
    return float(re.sub("[^0-9]+", "", string)) / 100

## Load data
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

data

## Plot distributions
sns.displot(
    data,
    x="damage taken",
    hue="encounter",
)

## Plot distributions
sns.displot(
    data,
    x="damage taken",
    row="encounter",
)

## Scatterplot
plt.figure(figsize=[10, 7])
sns.scatterplot(
    data=data,
    x="tod",
    y="damage taken",
    hue="encounter",
)
