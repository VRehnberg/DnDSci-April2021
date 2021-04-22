import numpy as np
import pandas as pd
from scipy import stats, optimize

from utils import load_data

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
    '''Model for the Grey Swans upcomming voyage.'''

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

        # Join everything together and shuffle
        sampled_data = pd.concat(sampled_data)
        sampled_data = sampled_data.sample(frac=1)
        sampled_data.reset_index(inplace=True)

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

    def make_voyage(
        self, 
        shark_repellant=False,
        woodworkers=False,
        merpeople_tribute=False,
        extra_oars=0,
        extra_cannons=0,
        arm_crows_nest=False,
        foam_swords=False,
        reps=1000,
    ):

        # Sample extra as sharks and merpeople are discarded
        voyages = self.sample(2*reps)

        # Removal interventions
        if shark_repellant:
            mask = voyages["encounter"]!="sharks"
            voyages = voyages[mask]

        if merpeople_tribute:
            mask = voyages["encounter"]!="merpeople"
            voyages = voyages[mask]

        # Multiplier interventions
        if woodworkers:
            mask = voyages["encounter"]=="crabmonsters"
            mult = np.where(mask, 0.5, 1.0)
            voyages["damage taken"] = mult * voyages["damage taken"]

        if extra_oars:
            if not (0 < extra_oars <= 20):
                raise ValueError(f"extra_oars is {extra_oars}.")
            mask = np.logical_or(
                voyages["encounter"]=="kraken",
                voyages["encounter"]=="demon whale",
            )
            mult = 1.0 - np.where(mask, extra_oars * 0.02, 0.0)
            voyages["damage taken"] = mult * voyages["damage taken"]

        if extra_cannons:
            if not (0 < extra_cannons <= 3):
                raise ValueError(f"extra_cannons is {extra_cannons}.")
            mask = np.logical_or(
                voyages["encounter"]=="nessie",
                voyages["encounter"]=="pirates",
            )
            mult = 1.0 - np.where(mask, extra_cannons * 0.1, 0.0)
            voyages["damage taken"] = mult * voyages["damage taken"]

        if arm_crows_nest:
            mask = voyages["encounter"]=="harpy"
            mult = np.full(voyages.shape[0], 1.0)
            mult[mask] = np.random.choice([0, 1], mask.sum(), p=[0.7, 0.3])
            voyages["damage taken"] = mult * voyages["damage taken"]
            
        if foam_swords:
            mask = voyages["encounter"]=="water elemental"
            mult = np.where(mask, 0.4, 1.0)
            voyages["damage taken"] = mult * voyages["damage taken"]

        # Reduce to reps number of voyages
        voyages = voyages[:reps]
        
        return voyages
        

if __name__=="__main__":
    
    
    model = Model()
    data = load_data()

    model.fit(data)

    print(model.sample(10))

    x = np.linspace(0, 1.5, 300)
    print(model.pdf(x).mean())

    ## Try some simulations

    reps = 10000

    damage_taken = model.make_voyage(reps=reps)["damage taken"].values
    frac_survived = (damage_taken < 1).sum() / len(damage_taken)
    print(f"No interventions => survival {frac_survived}")

    # Sanity check
    damage_taken = data["damage taken"].values
    frac_survived = (damage_taken < 1).sum() / len(damage_taken)
    print(f"Survival fraction from data is {frac_survived}")

    # All interventions
    damage_taken = model.make_voyage(
        shark_repellant=True,
        woodworkers=True,
        merpeople_tribute=True,
        extra_oars=20,
        extra_cannons=3,
        arm_crows_nest=True,
        foam_swords=True,
        reps=reps,
    )["damage taken"].values
    frac_survived = (damage_taken < 1).sum() / len(damage_taken)
    print(f"All interventions => survival {frac_survived}")

    # All but shark_repellant
    damage_taken = model.make_voyage(
        shark_repellant=False,
        woodworkers=True,
        merpeople_tribute=True,
        extra_oars=20,
        extra_cannons=3,
        arm_crows_nest=True,
        foam_swords=True,
        reps=reps,
    )["damage taken"].values
    frac_survived = (damage_taken < 1).sum() / len(damage_taken)
    print(f"All but shark_repellant => survival {frac_survived}")

    # All but shark_repellant
    damage_taken = model.make_voyage(
        shark_repellant=True,
        woodworkers=True,
        merpeople_tribute=False,
        extra_oars=20,
        extra_cannons=3,
        arm_crows_nest=True,
        foam_swords=True,
        reps=reps,
    )["damage taken"].values
    frac_survived = (damage_taken < 1).sum() / len(damage_taken)
    print(f"All but tribute => survival {frac_survived}")
