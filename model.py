import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize

from utils import load_data

PARAMETERS = {
    "counts" : 9,
    "crabmonsters": 2,
    "kraken": 2,
    "pirates": 5,
    "merpeople": 2,
    "harpy": 2,
    "water elemental": 2,
    "sharks": 2,
    "nessie": 2,
    "demon whale": 2,
}

class MixtureModel(stats.rv_continuous):
    '''Weighted mix of rv_continuous models.'''

    def __init__(self, submodels, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.weights = weights

    def _pdf(self, x):
        pdf = sum(
            weight * submodel.pdf(x)
            for weight, submodel
            in zip(self.weights, self.submodels)
        )
        return pdf

    def rvs(self, size):
        submodel_choices = np.random.choice(
            len(self.submodels),
            size=size,
            p=self.weights,
        )
        submodel_samples = [
            submodel.rvs(size=size)
            for submodel in self.submodels
        ]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs

class Model():
    '''Model for the Grey Swans upcomming journey.'''

    def __init__(self):
        self.submodels = {}

    def fit(self, all_data, verbose=True):
        '''Rough MLE fit under some assumptions.'''

        expected_counts = {}
        for encounter, data in all_data.groupby("encounter"):

            if encounter=="unknown":
                continue

            elif encounter=="pirates":
                # Pirates is a mix of a Gamma and a normal
                # also, the distributions changed historically
                tod_max = max(all_data["tod"].values)
                tod_threshold = 62.5
                mask = data["tod"]>tod_threshold
                x = data[mask]["damage taken"].values

                # Prepare for minimising neg log like
                p0 = [0.9, 10, 0.2, 0.4, 0.1]
                fun = lambda p: -np.log(
                    p[0] * stats.gamma(p[1], scale=p[2]/p[1]).pdf(p)
                    + (1 - p[0]) * stats.norm(p[3], p[4]).pdf(p)
                ).sum()
                bounds = [
                    (0.01, 0.99),
                    (1, 1000),
                    (0.1, 0.3),
                    (0.3, 0.5),
                    (0.01, 0.2),
                ]
                
                # Optimization
                res = optimize.minimize(fun, p0, bounds=bounds)
                p = res.x
                
                # Save model and expected counts
                self.submodels[encounter] = MixtureModel(
                    [
                        stats.gamma(p[1], scale=p[2]/p[1]),
                        stats.norm(p[3], p[4]),
                    ],
                    [p[0], 1 - p[0]], 
                )
                expected_count = round(x.size * tod_max / tod_threshold)
                expected_counts[encounter] = expected_count

                # Estimate num unused data from old distribution
                pirate_discard = len(data) - expected_count

            elif encounter=="water elemental":
                # Water elemental is a uniform distribution

                x = data["damage taken"].values

                p = stats.uniform.fit(x)
                self.submodels[encounter] = stats.uniform(*p)
                expected_counts[encounter] = len(data)

            else:

                # Rest are Gamma
                x = data["damage taken"].values
                n_x = x.size
            
                # Setup optimization
                p0 = (
                    [5, 0.5]
                    if not encounter=="crabmonsters"
                    else [1, 0.5]  # crabmonsters look exponential
                )
                fun = lambda p: -(
                    stats.gamma.logpdf(x, p[0], scale=p[1]/p[0])
                    - stats.gamma.logcdf(1, p[0], scale=p[1]/p[0])
                    + stats.norm.logpdf(# "prior" on expected counts
                        np.log(n_x / stats.gamma.cdf(1, p[0], scale=p[1]/p[0])),
                        np.log(2421),
                        np.log(1200),
                    )
                ).sum()
                bounds = [
                    (1, 1000),
                    (0.01, 5),
                ]

                # Optimize
                res = optimize.minimize(fun, p0, bounds=bounds)
                p = res.x
                submodel = stats.gamma(p[0], scale=p[1]/p[0])

                # Save results
                self.submodels[encounter] = submodel
                expected_count = round(n_x / submodel.cdf(1))
                expected_counts[encounter] = expected_count

        # Estimate multinomial parameters (counts)
        n = int(sum(expected_counts.values()))
        p = np.array(list(expected_counts.values())) / n
        self.submodels["counts"] = stats.multinomial(n, p)
        self.p_counts = p

        if verbose:
            print(f"Fitting complete.")
            print(f"Diff. expected vs actual counts ({n - len(all_data)}).")
            print(f"Pirate discard {pirate_discard}.")

    def sample(self, size):
        
        # Sample encounters
        p_counts = self.p_counts.copy()
        counts = stats.multinomial.rvs(
            size=1,
            n=size,
            p=p_counts,
        ).squeeze()

        sampled_data = []

        # Sample damage given encounter
        for count, (name, model) in zip(counts, self.submodels.items()):
            
            if count==0:
                continue

            data = pd.DataFrame({
                "damage taken": model.rvs(count),
                "encounter": np.full(count, name, dtype=object),
            })

            sampled_data.append(data)

        # Join everything together
        sampled_data = pd.concat(sampled_data)
        return sampled_data

    def pdf(self, x, encounters=None, renormalize=False):

        # None => All possible encounters
        all_encounters = list(self.submodels)
        if encounters is None:
            encounters = all_encounters

        # Select relevant models etc.
        included = [enc in encounters for enc in all_encounters]

        p_counts = self.p_counts.copy()
        if renormalize:
            p_counts = p_counts / p_counts.sum()

        models = [self.submodels[enc] for enc in encounters]

        # Calculate pdf
        pdf = sum(
            p_count * model.pdf(x)
            for p_count, model
            in zip(p_counts, models)
        )

        return pdf



if __name__=="__main__":
    
    
    model = Model()
    data = load_data()

    model.fit(data)

    print(model.sample(10))

    x = np.linspace(0, 1.5, 300)
    print(model.pdf(x).mean())
