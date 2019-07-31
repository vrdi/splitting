# Imports

import os
import geopandas as gpd
import numpy as np
import copy
import matplotlib.pyplot as plt
import networkx as nx
import pandas
import math

from gerrychain import (
    Graph, GeographicPartition, Partition, Election, accept, MarkovChain, constraints
)
from gerrychain.updaters import Tally, cut_edges
from gerrychain.constraints import (
    single_flip_contiguous, no_vanishing_districts
)
from gerrychain.proposals import propose_random_flip, recom
from gerrychain.accept import always_accept
from gerrychain.metrics import polsby_popper
from gerrychain.random import random

from collections import defaultdict, Counter
from itertools import combinations_with_replacement
from functools import partial

# setup -- SLOW

shapefile = "https://github.com/mggg-states/NC-shapefiles/raw/master/NC_VTD.zip"
print("shapefile loaded")
df = gpd.read_file(shapefile)
print("dataframe saved")

county_col = "County"
pop_col = "PL10AA_TOT"
uid = "VTD"

graph = Graph.from_geodataframe(df,ignore_errors=True)
print("made graph")
graph.add_data(df,list(df))
graph = nx.relabel_nodes(graph, df[uid])
counties = (set(list(df[county_col])))
countydict = dict(graph.nodes(data=county_col))

starting_partition = GeographicPartition(
    graph,
    assignment="judge",
    updaters={
        "polsby_popper" : polsby_popper,
        "cut_edges": cut_edges,
        "population": Tally(pop_col, alias="population"),

    }
)

county_edge_count = {}
for i in counties:
    county_graph = graph.subgraph([n for n,v in graph.nodes(data = True) if v[county_col] == i])
    total_edges = len(county_graph.edges())
    county_edge_count[i] = total_edges

def county_splits_dict(partition):
    """
    From a partition, generates a dictionary of counter dictionaries.

    Args: 
        partition: the partition for which a dictionary is being generated.

    Returns: 
        A dictionary with keys as dictrict numbers and values as Counter() dictionaries.
        These counter dictionaries have pairs County_ID: NUM which counts the number of
        VTDs in the county in the district. 
    """

    county_splits = {k:[] for k in counties}
    county_splits = {  k:[countydict[v] for v in d] for k,d in partition.assignment.parts.items()   }
    county_splits = {k: Counter(v) for k,v in county_splits.items()}
    return county_splits

def cut_edges_in_county(partition):
   """
   Computes a sum over all county scores, which are each calculated by
   taking the number of cut edges and dividing by the number of total edges.

   Args: 
        partition: the partition to be scored.

   Returns: 
        An integer score that is the sum of all county scores generated.
   """

   county_cut_edge_dict = {}
   cut_edge_set = partition["cut_edges"]
   for k in cut_edge_set:
       vtd_1 = k[0]
       vtd_2 = k[1]
       county_1 = countydict.get(vtd_1)
       county_2 = countydict.get(vtd_2)
       if county_1 == county_2:
           if county_1 in county_cut_edge_dict.keys():
               county_cut_edge_dict[county_1] += 1
           else:
               county_cut_edge_dict[county_1] = 1
   ratio_dict = {}
   for i in county_cut_edge_dict.keys():
       ratio = county_cut_edge_dict[i]/county_edge_count[i]
       ratio_dict[i] = ratio
   return sum(ratio_dict.values())

def cut_edges_in_district(partition):
    """
    Computes a ratio of the cut edges between two counties over the total number of cut edges.

    Args: 
        partition: the partition to be scored.

    Returns: 
        The ratio of cut edges between two counties over the total number of cut edges.
    """
    cut_edges_between = 0
    cut_edge_set = partition["cut_edges"]
    for i in cut_edge_set:
        vtd_1 = i[0]
        vtd_2 = i[1]
        county_1 = countydict.get(vtd_1)
        county_2 = countydict.get(vtd_2)
        if county_1 != county_2:
            cut_edges_between += 1
    num_cut_edges = len(cut_edge_set)
    score = cut_edges_between/num_cut_edges
    return score

def VTDs_to_Counties(partition):
    """
    Converts a partition into a dictionary with keys as districts and values as a list of VTDs in that district.

    Args: 
        partition: the partition to be converted.

    Returns:
        A dictionary with keys as districts and values as dictionaries of county-population key-value pairs. 
        This represents the population of each county that is in each district.
    """

    district_dict = dict(partition.parts)
    new_district_dict = dict(partition.parts)
    for district in district_dict.keys():
        vtds = district_dict[district]
        county_pop = {k:0 for k in counties}
        for vtd in vtds:
            county_pop[countydict[vtd]] += graph.nodes[vtd][pop_col]
        new_district_dict[district] = county_pop
    return new_district_dict

def dictionary_to_score(dictionary):
    """
    Calculates the half score of the Moon splitting score corresponding to the given dictionary.

    Args: 
        dictionary: the dictionary to be scored.

    Returns: 
        The half score of the dictionary.
    """

    district_dict = dictionary
    score = 0
    for dist in district_dict.keys():
        counties_and_pops = district_dict[dist]
        total = sum(counties_and_pops.values())
        fractional_sum = 0
        for county in counties_and_pops.keys():
            fractional_sum += np.sqrt(counties_and_pops[county]/total)
        score += total*fractional_sum
    return score

def invert_dict(dictionary):
    """
    Inverts a dictionary of dictionaries, switching the keys and values.

    Args:
        dictionary: dictionary to be inverted.

    Returns:
        An inverted dictionary.
    """

    new_dict = defaultdict(dict)
    for k,v in dictionary.items():
        for k2,v2 in v.items():
            new_dict[k2][k] = v2
    return new_dict
    
def moon_score(partition):
    """
    Calculates the Moon splitting score of a partition. 

    Args: 
        partition: the partition to be scored.

    Returns: 
        The Moon splitting score of the partition.

    """

    dictionary = VTDs_to_Counties(partition)
    return dictionary_to_score(dictionary) + dictionary_to_score(invert_dict(dictionary))

def mattingly(partition, M_C = 1000):
    """
    Calculates the Mattingly splitting score of a partition.

    Args: 
        partition: the partition to be scored.
        M_C: (default 1000).

    Returns: 
        The Mattingly splitting score of the partition.

    """

    num_2_splits = 0
    num_2_splits_W = 0
    num_greater_splits = 0
    num_greater_splits_W = 0
    county_splits = {k:[] for k in counties}
    dct = {  k:[countydict[v] for v in d] for k,d in partition.assignment.parts.items()   }
    dct = {k: Counter(v) for k,v in dct.items()}
    
    for v in dct.values():
        for k,ct in v.items():
            county_splits[k].append(ct)
                
    for county in county_splits.keys():
        if len( county_splits[county]) == 2:
            total = sum( county_splits[county])
            max_2 = min( county_splits[county])
            num_2_splits += 1
            num_2_splits_W += np.sqrt( max_2 / total )
        elif len(county_splits[county]) > 2:
            total = sum(county_splits[county])
            county_splits[county].sort()
            left_overs = total - county_splits[county][-1] - county_splits[county][-2]
            num_greater_splits += 1
            num_greater_splits_W += np.sqrt( left_overs / total)
    return num_2_splits * num_2_splits_W + M_C * num_greater_splits * num_greater_splits_W

def edge_entropy(partition):
    """


    Args: 
        partition: the partition

    Returns: 
        entropy
    """
    entropy = 0
    total_edges = len(graph.edges())
    countynodelist = {
        county: frozenset(
            [node for node in graph.nodes() if graph.nodes[node][county_col] == county]) for county in counties
    }
    districts_in_counties = {
        county: frozenset([partition.assignment[d] for d in countynodelist[county]]) for county in counties
    }
    for county in counties:
        county_subgraph = graph.subgraph([n for n in graph.nodes if graph.nodes[n][county_col] == county])
        county_edges = len(county_subgraph.edges())
        for (district1, district2) in combinations_with_replacement(districts_in_counties[county],2):
            p_ij = len([e for e in county_subgraph.edges() if set(
                [partition.assignment[e[0]], partition.assignment[e[1]]]) == set([district1, district2])])
            p_ij = p_ij/len(county_subgraph.edges())
            if (p_ij != 0):
                entropy -= p_ij*np.log(p_ij)*county_edges/total_edges
    return entropy

def num_of_splittings(partition):
    """
    Calculates the number of splittings in a partition.

    Args: 
        partition: the partition to be scored.

    Returns:
        The number of splittings in the partition.
    """
    dictionary = county_splits_dict(partition)
    counter = 0
    for district in dictionary.keys():
        counter += len(dictionary[district])
    return counter

#-------------------------------------------------------------------------------------------
totpop = 0
num_districts = 13
for n in graph.nodes():
    graph.node[n][pop_col] = int(graph.node[n][pop_col])
    totpop += graph.node[n][pop_col]

proposal = partial(
        recom, pop_col=pop_col, pop_target=totpop/num_districts, epsilon=0.02, node_repeats=1
    )

compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]), 2 * len(starting_partition["cut_edges"])
    )

chain = MarkovChain(
        proposal,
        constraints=[
            constraints.within_percent_of_ideal_population(starting_partition, 0.05),compactness_bound
          #constraints.single_flip_contiguous#no_more_discontiguous
        ],
        accept=accept.always_accept,
        initial_state=starting_partition,
        total_steps=5000
    )

cuts = []
splittings = []
moon_metric_scores = [] #CHANGE


t = 0
for part in chain:
    cuts.append(len(part["cut_edges"]))
    splittings.append(num_of_splittings(part))
    moon_metric_scores.append(moon_score(part)) #CHANGE
    
    t += 1
    if t % 100 == 0:
        print("finished chain " + str(t))
        
np.savetxt("NC_cuts.txt", cuts)
np.savetxt("NC_splittings.txt", splittings)
np.savetxt("NC_symmetric_entropy_moon.txt", moon_metric_scores) #CHANGE

#CHANGE
plt.figure()
plt.hist(moon_metric_scores)
plt.title("Histogram of Symmetric Entropy (Moon)")
plt.xlabel("Symmetric Entropy")
plt.savefig("NC_hist_symmetric_entropy_5000.png")
plt.show()

#CHANGE
plt.figure()
plt.scatter(splittings, moon_metric_scores)
plt.title("Symmetric Entropy (Moon) vs. Number of Splits")
plt.xlabel('Splits')
plt.ylabel('Symmetric Entropy')
plt.xlim(min(splittings) - 5, max(splittings) + 5)
plt.savefig("NC_scatter_symmetric_entropy_splits_5000.png")
plt.show()
#CHANGE
plt.figure()
plt.scatter(cuts, moon_metric_scores)
plt.title("Symmetric Entropy (Moon) vs. Number of Cut Edges")
plt.xlabel('Cut Edges')
plt.ylabel('Symmetric Entropy')
plt.savefig("NC_scatter_symmetric_entropy_cut_edges_5000.png")
plt.show()




#print(moon_score(starting_partition))
#print(list(starting_partition.assignment.keys()))
#print([n for n in graph.nodes if n not in starting_partition.assignment])
#print(df["VTD"])
#print(starting_partition["cut_edges"])


#d = county_splits_dict(starting_partition)
#print(cut_edges_in_county(starting_partition))
#print(cut_edges_in_district(starting_partition))
#sum( [ len([ dd for dd  in [dict(v) for v in d.values()] if k in dd.keys()]) > 1 for k in counties]    )