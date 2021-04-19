# This file is written for jupyter-vim

## Imports
import re
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

from utils import load_data

sns.set_context("talk")

## Load data
data = load_data()

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

## Plot distributions
sns.displot(
    data,
    x="damage taken",
    row="encounter",
)
