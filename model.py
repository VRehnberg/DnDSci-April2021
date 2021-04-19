import numpy as np
import emcee
from scipy import stats

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


def log_gamma_param_prior(a, b):
    "p(a, µ) \propto (1 / (aµ))"
    mean = a / b
    if not (1 <= a <= 1000) or not (0.1 <= mean <= 5):
        return -np.inf
    else:
        return -a - b

def log_uniform_param_prior(start, width):
    '''p(width) \propto 1 / width'''
    if not (0 <= start <= 1) or not (0.1 < width <= 1):
        return -np.inf
    else:
        return -width

def log_prior(params):
    '''Calculates the log prior for the parameters.'''

    # Initialize relevant values
    names, n_params = zip(*PARAMETERS.items())
    split_params = np.array_split(params, np.cumsum([n_params]))
    log_prior = 0.0

    # Inner priors
    for name, n, p in zip(names, n_params, split_params):
        
        if (name=="counts"):
            if sum(p)!=1.0 or any(p<0.0):
                return -np.inf
            log_prior += stats.dirichlet(np.ones(n)).logpdf(p)

        elif (name=="pirates"):
            (p_inner, a1, b1, a2, b2) = p

            log_prior += (
                stats.beta(1, 1).logpdf(p_inner)
                + log_gamma_param_prior(a1, b1)
                + log_gamma_param_prior(a2, b2)
            )

        elif (name=="water elemental"):
            (start, width) = p

            log_prior += log_uniform_param_prior(start, width)

        else:
            (a, b) = p
            log_prior += log_gamma_param_prior(a, b)

    return log_prior


def log_likelihood(damage_data, params):

    # Interpret relevant values
    names, n_params = zip(*PARAMETERS.items())
    split_params = np.array_split(params, np.cumsum([n_params]))
    p_counts = split_params[0] 

    # Initialize likelihoods
    log_like = 0.0
    p_sunk = 0.0


    for name, p_count, param in zip(
        names[1:],
        p_counts,
        split_params[1:]
    ):

        # Calculate likelihood
        mask = (damage_data["encounter"]==name)
        x = damage_data[mask]["damage taken"].values
        n_x = x.shape[0]

        log_like += n_x * p_count

        if (name=="pirates"):
            # TODO pirates only of time greater than something
            (p_inner, a1, b1, a2, b2) = param
            log_like += np.log(
                p_inner * stats.gamma(a1, scale=1/b1).pdf(x)
                + (1 - p_inner) * stats.gamma(a2, scale=1/b2).pdf(x)
            ).sum()
            p_sunk += p_count * (
                p_inner * stats.gamma(a1, scale=1/b1).sf(1)
                + (1 - p_inner) * stats.gamma(a2, scale=1/b2).sf(1)
            )

        elif (name=="water elemental"):
            (s, w) = param
            log_like += stats.uniform(s, w).logpdf(x).sum()
            p_sunk += p_count * stats.uniform(s, w).sf(1)

        else:
            (a, b) = param
            log_like += stats.gamma(a, scale=1/b).logpdf(x).sum()
            p_sunk += p_count * stats.gamma(a, scale=1/b).sf(1)

    # Sunk
    mask = (damage_data["encounter"]=="unknown")
    n_x = sum(mask)
    log_like += n_x * p_sunk

    return log_like

def log_posterior(params, damage_data):
    '''Computes the log posterior of the parameters.'''

    log_posterior = 0.0

    log_posterior += log_prior(params)
    log_posterior += log_likelihood(damage_data, params)

    return log_posterior

if __name__=="__main__":
    
    # Test log probabilities
    n = 1
    test_params = np.hstack([
        stats.dirichlet(np.ones(9)).rvs(n),
        np.hstack([
            (
                np.random.rand(n, n_par) + 1
                if name!="water elemental"
                else np.sort(np.random.rand(n, n_par))
            ) if name!="pirates"
            else np.hstack([
                np.random.rand(n, 1),
                np.random.rand(n, n_par - 1) + 1,
            ])
            for name, n_par in PARAMETERS.items()
            if name!="counts"
        ]),
    ]).T.squeeze()

    data = load_data()

    print(log_prior(test_params))
    print(log_likelihood(data, test_params))
    print(log_posterior(test_params, data))

    # Try sampling
    nwalkers = 100
    ndim = sum(PARAMETERS.values())
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data])

    n = nwalkers
    p0 = np.hstack([
        stats.dirichlet(np.ones(9)).rvs(n),
        np.hstack([
            (
                np.random.rand(n, n_par) + 1
                if name!="water elemental"
                else np.sort(np.random.rand(n, n_par))
            ) if name!="pirates"
            else np.hstack([
                np.random.rand(n, 1),
                np.random.rand(n, n_par - 1) + 1,
            ])
            for name, n_par in PARAMETERS.items()
            if name!="counts"
        ]),
    ])
    n_samples = 100
#     sampler.run_mcmc(p0, 100)

    
