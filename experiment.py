## Imports
import re
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
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

## Summaries
data.describe()

## Plot distributions
sns.displot(
    data,
    x="damage taken",
    hue="encounter",
)

## Time series

fgrid = sns.relplot(
    data=data,
    x="tod",
    y="damage taken",
    row="encounter",
    hue="direction",
    sizes=1,
    alpha=0.1,
)
vline = lambda *args, **kwargs: plt.axvline(61.5, c="r")
fgrid.map(vline)
    

##

fgrid = sns.relplot(
    data=data,
    x="tod",
    y="damage taken",
    row="encounter",
    hue="direction",
    style="direction",
    kind="line",
)
vline = lambda *args, **kwargs: plt.axvline(61.5, c="r")
fgrid.map(vline)

##

fgrid = sns.relplot(
    data=data.groupby(["tod", "direction", "encounter"])["damage taken"].describe(),
    x="tod",
    y="count",
    row="encounter",
    hue="direction",
    style="direction",
    kind="line",
)
vline = lambda *args, **kwargs: plt.axvline(62.5, c="r")
fgrid.map(vline)

##

sns.relplot(
    data=data,
    x="month",
    y="damage taken",
    row="encounter",
    hue="direction",
    style="direction",
    kind="line",
)

##

sns.relplot(
    data=data.groupby(["month", "direction", "encounter"])["damage taken"].describe(),
    x="month",
    y="count",
    row="encounter",
    hue="direction",
    style="direction",
    kind="line",
)

## Plot distribution
encounter = "pirates"
pirate_data = data[data["encounter"]==encounter].copy()
pirate_data["before threshold"] = pirate_data["tod"] < 62.5
sns.displot(
    pirate_data,
    x="damage taken",
    hue="before threshold",
)

## Plot distribution
encounter = "demon whale"
mask = data["encounter"]==encounter
sns.displot(
    data[mask],
    x="damage taken",
    hue="before threshold",
)
