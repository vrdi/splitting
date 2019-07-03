## Scores

__Mattingly__
$$J_C(\mathrm{Plan}) = (\# \text{counties split between two districts})\cdot W_{2} + M_C\cdot (\# \text{counties split between three or more districts})\cdot W_{3}$$

$M_C$ is a large constant

$$W_2 = \sum\limits_{\text{counties split between two districts}} (\text{fraction of county VTDs in 2nd largest intersection of a district with a county})$$


$$W_3 = \sum\limits_{\text{counties split between two districts}} (\text{fraction of county VTDs not in the largest or second largest intersections of a  district with a county})^{\frac{1}{2}}$$

- penalizes 50-50 worse than 99-1
- feels tuned to NC which has no massive counties (only Wake and Mecklenburg are big)

- + simple and easy to implement
- + it's clear what it's measuring
- + somewhat smoothly transitions from no splits to tiny splits
- - feels a little arbitrary
- - agnostic to geography


__Pennsylvania__
Rule: Any county can't contain more congressional districts than the number required plus one (plus one for state senate, plus two for state house).  You don't have to draw whole districts inside counties. You shouldn't split unless necessary for population balance

- + simple to state
- + considers the populations of counties
- - unclear about rules for splitting
- - doesn't discern between one bad split and lots of bad splits
- - doesn't see geography/compactness


__VA Criteria__
Localities denoted $L_j$ with population $\ell_j$. Districts are $D_i$ with population $d_i$.  $P_{j}^{D_i}$ is the fraction of $d_i$ in $L_j$ and $P_{i}^{L_j}$ is the fraction of $\ell_j$ contained in $D_i$.

$$\mathrm{Split} = \sum\limits_{i} d_i [ \sum\limits_j \sqrt{P_{j}^{D_i}}] + \sum\limits_j \ell_j [\sum\limits_i \sqrt{P_{i}^{L_j}}]$$

- Entropy-flavored

- + has a symmetry
- + takes into account the idea that some localities contain districts and some districts contain localities
- - doesn't see geography



__Conditional Entropy__
$$\mathrm{Ent}(D\vert C) = \sum\limits_j q_j \sum\limits_i p_{i\vert j}\log{\frac{1}{p_{i\vert j}}}$$
 
 Where $p_{i\vert j}$ is the proportion of county $j$ which is occupied by district $i$.


 __Splittings 1.0__
 It's conditional entropy, but with a power between 0 and 1 instead of the logarithm. They do this to get around the proportional refinement property of log which doesn't penalize something like 98-1-1 much worse than 98-2, whereas they want the first to be way worse than the second. They also switch the $q_j$ for $1/q_j$ to penalize splitting small counties more.


- + their "good properties" are good, and this measure satisfies them
 - - doesn't see geography