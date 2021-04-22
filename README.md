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

## Submitted comment
I started by looking at the data and I figured that the distributions probably were quite nice. From looking at the data there seem to be nothing especially weird going on with direction, month or time. There are a few notable things that are a bit less nice than the rest.

Pirates are bimodal and there one of modes are reduced in intensity after approximately 62 weeks. Water elementals are uniform, while most of the rest looks like they could be captured by Gamma distribution. As such I assumed that I could model everything as a Gamma distribution, except pirates that were a Binomial over two Gamma and water elementals that were uniform and then a Multinomial for what encounter.

At first I tried a full Bayesian approach to model for the parameters but MC sampling this got a bit troublesome and I didn't want to put in the time to do it properly. Instead I did a rough MLE estimate were I assumed independence which technically didn't hold as they share the unknown data and I have around 700 data points from the unknown labels that were unallocated. Nevertheless, I now had something I could use to search for the best solution.

Trying it out it doesn't look good, using all the interventions leads to something that has around 3 % chance of surviving the cumulative damage from ten constitutive trips. If this is the case I would try very hard not to go.

The best attempt I found (though the rate was so small so I didn't get a good certainty for this) was to use shark repellant, pay the merpeople a tribute and buy 15 extra oars for the rest of the gold. The result was then

* Chance of survival 0.25 %
* Average damage on surviving ships 77 %
* Gold left 0

I haven't checked this though, so there might very well be some bug that gives me wrong results. These interventions were contrary to what I expected at the start. Then I was thinking about how the reduce the certain death encounters and that low and medium risk encounters were good. This intervention seems to be to increase the chance of the smallest risk encounters (even it this increases high risk encounters as well) and hope for the best.

## Ship is fixed at every stop!
Apparently I didn't read the instructions properly. The actual results of the model gets best results from buyin
 * Woodworker weapons
 * Tribute to seapeople
 * 20 oars (19 according to model but this is probably from randomness)
 * 1 cannon
the results are
 * 95 % survival for the collection of all ten trips
 * 41 % damage taken on average per trip
 * 5 gold remaining

Also, the DM has released his model [here](https://github.com/H-B-P/d-and-d-sci-apr/blob/main/gen.py).
