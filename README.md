# DnDSci: Voyage of the Grey Swan
This is my attempt at a [problem](https://www.lesswrong.com/posts/S3LKfRtYxhjXyWHgN/d-and-d-sci-april-2021-voyages-of-the-gray-swan) by [abstractapplic](https://www.lesswrong.com/users/abstractapplic).

## Initial ideas
 * The dangers that are registered are those that are not always letal.
 * The data probably follows some not too horible distributions, try to fit these distributions to get a model that can be used in MC method.
 * Patterns can change with how the direction of the voyage.
 * Patterns can be cyclic through the seasons.
 * Patterns can change over the years.

## Models
From looking at the data I think that I can model damage taken as Gamma distributions for the most part. For the most part time or direction doesn't seem to matter noticably.

Noteable possible exceptions:
 * Crabmonsters have a very large spread.
 * Pirates seem to be two combined distributions where one of them had a significantly reduced damage output in the later ~half
 * Water elemental seems to be uniform around 0.8±0.5.
 * Unknown are probably divided among the distributions that sometimes exceed 1.

