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

def log_gamma_parame_prior(a, b):
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
    split_param = np.array_split(params, np.cumsum([n_params]))
    log_prior = 0.0

    # Inner priors
    for name, n, p in zip(names, n_prams, split_params):
        
        if (name=="counts"):
            log_prior += stats.dirichlet(np.ones(n)).logpdf(p)

        elif (enc=="pirates"):
            (p_inner, a1, b1, a2, b2) = p

            log_prior += (
                stats.beta(1, 1).logpdf(p_inner)
                + log_gamma_param_prior(a1, b1)
                + log_gamma_param_prior(a2, b2)
            )

        elif (enc=="water elemental"):
            (start, width) = p

            log_prior += log_uniform_param_prior(start, width)

        else:
            (a, b) = p
            log_prior += log_gamma_param_prior(a, b)

    return log_prior


def log_likelihood(damage_data, params):

    # Initialize relevant values
    names, n_params = zip(*PARAMETERS.items())
    split_param = np.array_split(params, np.cumsum([n_params]))
    log_like = 0.0

    p_counts = split_param[0] 

    for name, p_count, param in zip(
        names[1:],
        p_counts,
        split_params[1:]
    ):

        # Calculate likelihood
        mask = (damage_data["encounter"]==name)
        x = damage_data[mask]["damage taken"].values

        log_like += p_count

        if (name=="pirates"):
            (p_inner, a1, b1, a2, b2) = param
            log_like += (
                p_inner + stats.gamma(a1, scale=1/b1).logpdf(x)
                + (1 - p_inner) + stats.gamma(a2, scale=1/b2).logpdf(x)
            )

        elif (name=="water elemental"):
            (s, w) = param
            log_like += stats.uniform(s, w).logpdf(x)

        else:
            (a, b) = param
            log_like += stats.gamma(a, b).logpdf(x)

    return log_like


def log_posterior(params, damage_data):
    '''Computes the log posterior of the parameters.'''

    log_posterior = 0.0

    log_posterior += log_prior(params)
    log_posterior += log_likelihood(damage_data, params)

    return log_posterior
