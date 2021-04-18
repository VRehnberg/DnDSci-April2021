ENCOUNTERS = [
    "crabmonsters",
    "kraken",
    "pirates",
    "merpeople",
    "harpy",
    "elemental",
    "sharks",
    "nessie",
    "whale",
]

def log_prior(params):
    '''Calculates the log prior for the parameters.

    Arguments:
        params (list): The parameters are [
            p_counts,
            (a_crabs, b_crabs),
            (a_kraken, b_kraken),
            p_pirates,
            (a1_pirates, b1_pirates),
            (a2_pirates, b2_pirates),
            (a_merpeople, b_merpeople),
            (a_harpy, b_harpy),
            (s_elemental, w_elemental),
            (a_sharks, b_sharks),
            (a_nessie, b_nessie),
            (a_whale, b_whale),
    ]
    '''

    log_prior = 0

    (
        p_counts,
        (a_crabs, b_crabs),
        (a_kraken, b_kraken),
        p_pirates,
        (a1_pirates, b1_pirates),
        (a2_pirates, b2_pirates),
        (a_merpeople, b_merpeople),
        (a_harpy, b_harpy),
        (s_elemental, w_elemental),
        (a_sharks, b_sharks),
        (a_nessie, b_nessie),
        (a_whale, b_whale),
    ) = np.array_split(params, np.cumsum([
        9, # which encounter
        2, # gamma crabs
        2, # gamma kraken
        1, # which pirates
        2, # gamma pirates 1
        2, # gamma pirates 2
        2, # gamma merpeople
        2, # gamma harpy
        2, # flat  water elemental
        2, # gamma sharks
        2, # gamma nessie
        2, # gamma demon whale
    ])

    # Gamma
    a_gamma = np.array([
        a_crabs,
        a_kraken,
        a1_pirates,
        a2_pirates,
        a_merpeople,
        a_harpy,
        a_sharks,
        a_nessie,
        a_whale,
    ])
    b_gamma = np.array([
        b_crabs,
        b_kraken,
        b1_pirates,
        b2_pirates,
        b_merpeople,
        b_harpy,
        b_sharks,
        b_nessie,
        b_whale,
    ])
    mean_gamma = a_gamma / b_gamma

    # Check damage distribution parameter ranges
    if any(np.reduce.logical_or([
        (0.1 > a_gamma).any(), (a_gamma > 1000).any(),
        (0.1 > mean_gamma).any(), (mean_gamma > 5).any(),
        0 > s_elemental, s_elemental > 1,
        w_elemental > 1,
    ])):
        return -np.inf

    # Counts
    log_prior += stats.dirichlet(np.ones_like(p_counts)).logpdf(p_counts)
    
    # Pirate binomial
    log_prior += stats.beta(1, 1).logpdf(p_pirates)

    # Gammas
    log_prior += np.sum(1 / a_gamma)
    log_prior += np.sum(1 / mean_gamma)

    # Uniform
    

    return log_prior

def log_likelihood(damage_data, params):

    log_like = 0.0

    # Interpret damage_data
    mask = damage_data["encounter"] == "crabmonsters"
    data_crabs = damage_data[mask]["damage_taken"].values

    mask = damage_data["encounter"] == "kraken"
    data_kraken = damage_data[mask]["damage_taken"].values

    mask = damage_data["encounter"] == "pirates"
    data_pirates = damage_data[mask]["damage_taken"].values

    mask = damage_data["encounter"] == "merpeople"
    data_merpeople = damage_data[mask]["damage_taken"].values

    mask = damage_data["encounter"] == "harpy"
    data_harpy = damage_data[mask]["damage_taken"].values

    mask = damage_data["encounter"] == "water elemental"
    data_elemental = damage_data[mask]["damage_taken"].values

    mask = damage_data["encounter"] == "sharks"
    data_sharks = damage_data[mask]["damage_taken"].values

    mask = damage_data["encounter"] == "nessie"
    data_nessie = damage_data[mask]["damage_taken"].values

    mask = damage_data["encounter"] == "demon whale"
    data_whale = damage_data[mask]["damage_taken"].values

    # Interpret params
    (
        p_counts,
        (a_crabs, b_crabs),
        (a_kraken, b_kraken),
        p_pirates,
        (a1_pirates, b1_pirates),
        (a2_pirates, b2_pirates),
        (a_merpeople, b_merpeople),
        (a_harpy, b_harpy),
        (s_elemental, w_elemental),
        (a_sharks, b_sharks),
        (a_nessie, b_nessie),
        (a_whale, b_whale),
    ) = np.array_split(params, np.cumsum([
        9, # which encounter
        2, # gamma crabs
        2, # gamma kraken
        1, # which pirates
        2, # gamma pirates 1
        2, # gamma pirates 2
        2, # gamma merpeople
        2, # gamma harpy
        2, # flat  water elemental
        2, # gamma sharks
        2, # gamma nessie
        2, # gamma demon whale
    ])

    (
        p_crabs,
        p_kraken,
        p_pirates,
        p_merpeople,
        p_harpy,
        p_elemental,
        p_sharks,
        p_nessie,
        p_whale,
    ) = p_counts

    #

    # Crabs
    x = data_crabs
    n_x = data_crabs.shape[0]
    log_like += n_x * p_crabs + stats.gamma(a_crabs, scale=1/b_crabs).logpdf(x)
    
    # Kraken
    x = data_kraken
    log_like += p_kraken * stats.gamma(a_kraken, scale=1/b_kraken).logpdf(x)
    
    # Pirates
    x = data_pirates
    log_like += p_pirates * (
        p_pirates1 * stats.gamma(a_pirates, scale=1/b_pirates).logpdf(x)

    # Merpeople
    x = data_merpeople
    log_like += p_merpeople * stats.gamma(a_merpeople, scale=1/b_merpeople).logpdf(x)

    # Harpy
    x = data_harpy
    log_like += p_harpy * stats.gamma(a_harpy, scale=1/b_harpy).logpdf(x)

    # Elemental
    x = data_elemental
    log_like += p_elemental * stats.uniform(s_elemental, w_elemental).logpdf(x)

    # Sharks
    x = data_sharks
    log_like += p_sharks * stats.gamma(a_sharks, scale=1/b_sharks).logpdf(x)

    # Nessie
    x = data_nessie
    log_like += p_nessie * stats.gamma(a_nessie, scale=1/b_nessie).logpdf(x)

    # Whale
    x = data_whale
    log_like += p_whale * stats.gamma(a_whale, scale=1/b_whale).logpdf(x)

    
    

def log_posterior(params, damage_data):
    '''Computes the log posterior of the parameters.'''

    log_posterior = 0.0

    log_posterior += log_prior(params)
    log_posterior += log_likelihood(damage_data, params)

    return log_posterior
