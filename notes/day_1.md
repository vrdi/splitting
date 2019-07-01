__Why is splitting bad?__
- counties and towns as communities
- limits the ability to gerrymander
- reduces the complexity of the problem
	- preserving counties reduces the number of weird things you can do
	- "don't split" is very clear, but "some splitting allowed" has a lot of leeway
- can be hard to describe weird boundaries
	- important to know who represents you and for reps to know constituents
- it's nice when towns/counties/districts nest
- county lines don't change

__How can we measure splitting (Take One)?__
- count the number of split counties
- length of the split line vs as-the-crow-flies distance
- proportion of in-county edges which are cut
- ratio of cut edges leaving the county to cut edges in the county
- count the number of pieces a county is split into
- look at the number of pieces of the county which belong to each district
- area vs perimeter of the portion of a district in a county

__Properties of a good measure (Take One)__
- not splitting should be better than splitting
- should be worse with the number of splits (50-50 is "better" than "25-25-25-25")
- should take into account the "importance" of a unit
	- county splits in Maryland is worse than Massachusetts because the county is more meaningful there
	- at the least we should be able to tune the measure for this, maybe different measures for different states
	- the context of communities is important
- should generally point in the same direction as compactness
	- doing "bizzare" things to counties should be penalized
  
  
  
__Score 1: Ratio of cut edges in a county to number of edges in the county__
- assigns a score to a county
- only looking at edges inside of a county (not edges between counties)
- score approaches 1 as the number of cut edges grows (i.e. county in little bits is bad)
- score is 0 when county is intact
- all else fixed, a longer boundary (more cut edges) is worse
- we can't distinguish between the sizes of the chunks or number of chunks, only the lengths of the boundaries
- obeys refinement (i.e. adding more cutting is strictly worse)
- has a compactness flavor via cut edges

__Score 2: ratio of cut edges within a county to all cut edges__
- assigns a score to a district
- measures the coincidence of a district boundary with county boundary
- equal to 0 if all cut edges are county boundary (i.e. district doesn't split counties)
- equal to 1 when there is no coincidence between county and district boundaries
- feels dual to Score 1


__Score 3: sum of proportions of edges cut for each district touching a county__
- penalizes cuts into small chunks (i.e. 99-1 is worse than 50-50 for a fixed cut length)


__How do we aggregate?__
- we have things that assign scores to counties/districts, how do we get a score for a plan?
- population weighting (penalize splitting large or small counties)?
- normalization?
