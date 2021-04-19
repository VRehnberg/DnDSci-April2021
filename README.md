# DnDSci: Voyage of the Grey Swan
This is my attempt at a [problem](https://www.lesswrong.com/posts/S3LKfRtYxhjXyWHgN/d-and-d-sci-april-2021-voyages-of-the-gray-swan) by [abstractapplic](https://www.lesswrong.com/users/abstractapplic).

This README file can be seen as rough log over my process of investigation if read from top to bottom.

## Initial ideas
 * The dangers that are registered are those that are not always letal.
 * The data probably follows some not too horible distributions, try to fit these distributions to get a model that can be used in MC method.
 * Patterns can change with how the direction of the voyage.
 * Patterns can be cyclic through the seasons.
 * Patterns can change over the years.

## Looking at the data
From looking at the data I think that I can model damage taken as Gamma distributions for the most part. For the most part time or direction doesn't seem to matter noticably.

Noteable possible exceptions:
 * Crabmonsters have a very large spread.
 * Pirates seem to be two combined distributions where one of them had a significantly reduced damage output in the later ~half
 * Water elemental seems to be uniform around 0.8±0.5.
 * Unknown are probably divided among the distributions that sometimes exceed 1.

## Probabilisitic model
Meta-"prior":
 * As the name of the boat is [grey swan](https://accendoreliability.com/black-swans-grey-swans-white-swans/) I guess that I can disregard possibilities that some of the unknown causes are black swans (and it becomes slightly easier to model).
 * The samples are probably drawn using some common simple parametric probability distributions.
 * Looking at the data it seems that most of the damage distributions for a given encounter could be modeled as a Gamma distribution (though water elementals seems to be uniform and for pirates I'll use two Gamma distributions and a Binomial between them).
 * The counts of different encounters is (probably) Multinomial.

To be able to make predictions for how certain interventions work I want to find the parameters to the probability distributions.

Prior:
 * $\mathrm{Dirichlet}(1, 1, ...)$ for the probabilities over the possible encounters.
 * For the gamma $p\left(\mu=\frac{\alpha}{\beta}\right) \propto 1/\mu$ for $\mu \in [0.1, 5]$ and $p(\alpha) \propto 1/\alpha$ for $\alpha \in [1, 1000]$.
 * Beta(1, 1) for the two pirate distributions.

Likelihoods:
 * Multinomial for count of encounters times $p(\mathrm{damage} \in [0, 1)| \mathrm{encounter})$.
 * Gamma for damage received.
 * Note: For the unknown encounters the probability is the marginal probability $p(\mathrm{damage} \geq 1)$.
 * Pirates have two gamma with Binomial for which is used for each pirate encounter, these will have to be marginalised.
 
## Updating probabilistic model
After constructing the log-posterior for the probabilistic model I found it to be a hassle to sample from it with MCMC. Instead I simplified it so that I now do an MLE for one encounter at the time, with some quite ad hoc extra priors (so kinda a MAP).
