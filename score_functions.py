# Imports
import os
from gerrychain import Graph, GeographicPartition, Partition, Election, accept
from gerrychain.updaters import Tally, cut_edges
import geopandas as gpd
import numpy as np
from gerrychain.random import random
import copy
import seaborn as sns

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.accept import always_accept
from gerrychain.metrics import polsby_popper
from gerrychain import constraints
from gerrychain.constraints import no_vanishing_districts

from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx
import pandas
import math
from itertools import combinations_with_replacement
from functools import partial

def locality_splits_dict(partition, graph, locality_col, df):
    '''
    From a partition, generates a dictionary of counter dictionaries.

    :param partition: The partition for which a dictionary is being generated.
    :param locality_col: The string of the locality column's name. 
    :param df: The dataframe.

    :return: 1) A dictionary with keys as dictrict numbers and values as Counter() dictionaries. These counter dictionaries have pairs County_ID: NUM which counts the number of VTDs in the county in the district. 2) A set of the localities in a state.
    '''
    localitydict = dict(graph.nodes(data=locality_col))
    localities = (set(list(df[locality_col])))

    locality_splits = {k:[] for k in localities}
    locality_splits = {k:[localitydict[v] for v in d] for k,d in partition.assignment.parts.items()}
    locality_splits = {k: Counter(v) for k,v in locality_splits.items()}
    return locality_splits, localities

def num_parts(partition, graph, locality_col, df):
    '''
    Calculates the number of unique county-district pairs.

    :param partition: The partition to be scored.
    :param locality_col: The string of the locality column's name. 
    :param df: The dataframe of the state shapefile.

    :return: The number of parts, i.e. the number of unique county-district pairs.
    '''
    locality_splits, localities = locality_splits_dict(partition, graph, locality_col, df)

    counter = 0
    for district in locality_splits.keys():
        counter += len(locality_splits[district])
    return counter

def coincident_boundaries(partition, graph, locality_col):
    '''
    Computes a ratio of the cut edges between two counties over the total number of cut edges.

    :param partition: The partition to be scored.
    :param graph: The graph of the state shapefile. 

    :return: The ratio of cut edges between two counties over the total number of cut edges.
    '''
    countydict = dict(graph.nodes(data=locality_col))

    cut_edges_within = 0
    cut_edge_set = partition["cut_edges"]
    for i in cut_edge_set:
        vtd_1 = i[0]
        vtd_2 = i[1]
        county_1 = countydict.get(vtd_1)
        county_2 = countydict.get(vtd_2)
        if county_1 == county_2: #not on county boundary
            cut_edges_within += 1
    num_cut_edges = len(cut_edge_set)
    return cut_edges_within

def vtds_per_district(locality_splits):
    """
    A function that counts the number of VTDs per district.
    
    :param locality_splits: A dictionary with keys as district numbers and values Counter() dictionaries. The Counter dictionaries have pairs COUNTY_ID: NUM which counts the number of VTDS in the county in the district.
        
    :return: The total number of vtds in a district.
    """
    
    vtds = {}
    
    for district in locality_splits.keys():
        sum = 0
        counter = locality_splits[district]
        for vtd in counter.values():
            sum += vtd
        vtds[district] = sum
    return vtds 

def shannon_entropy(partition, graph, locality_col, df):
    '''
    Computes the shannon entropy score of a district plan.
    
    :param partition: The partition to be scored.
    :param locality_col: The string of the locality column's name. 
    :param df: A dataframe of the state shapefile.

    :returns: Shannon entropy score.
    '''
    locality_splits, localities = locality_splits_dict(partition, graph, locality_col, df)
    vtds = vtds_per_district(locality_splits) 
    num_districts = len(locality_splits.keys())

    total_vtds = 0
    for k,v in locality_splits.items():
        for x in list(v.values()):
            total_vtds += x

    entropy = 0
    for locality_j in localities:         #iter thru localities to get total count
        tot_county_vtds = 0
        #iter thru counters
        for k,v in locality_splits.items():
            v = dict(v)
            if locality_j in list(v.keys()):
                tot_county_vtds += v[locality_j]
            else:
                continue
        
        inner_sum = 0
        q = tot_county_vtds / total_vtds
        
        #iter thru districts to get vtds in county in district
        for district in range(num_districts):            
            counter = dict(locality_splits[district+1])            
            if locality_j in counter:
                intersection = counter[str(locality_j)]
                p = intersection / tot_county_vtds

                if p != 0:
                    inner_sum += p * math.log(1/p)
            else: 
                continue
        entropy += q * (inner_sum)
    return entropy


def power_entropy(partition, graph, locality_col, df, alpha):

    '''
    Computes the power entropy score of a district plan.
    
    :param partition: The partition to be scored.
    :param locality_col: The string of the locality column's name. 
    :param df: A dataframe of the state shapefile.
    :param alpha: A value between 0 and 1. 

    :returns: Power entropy score.
    '''

    locality_splits, localities = locality_splits_dict(partition, graph, locality_col, df)
    vtds = vtds_per_district(locality_splits) 
    num_districts = len(locality_splits.keys())

    total_vtds = 0              #count the total number of vtds in state
    for k,v in locality_splits.items():
        for x in list(v.values()):
            total_vtds += x

    entropy = 0
    for locality_j in localities:         #iter thru localities to get total count
        tot_county_vtds = 0
        #iter thru counters
        for k,v in locality_splits.items():
            v = dict(v)
            if locality_j in list(v.keys()):
                tot_county_vtds += v[locality_j]
            else:
                continue
        
        inner_sum = 0
        q = tot_county_vtds / total_vtds
        
        #iter thru districts to get vtds in county in district
        for district in range(num_districts):            
            counter = dict(locality_splits[district+1])            
            if locality_j in counter:
                intersection = counter[str(locality_j)]
                p = intersection / tot_county_vtds

                if p != 0:
                    inner_sum += p ** (1-alpha)
            else: 
                continue
        entropy += 1/q * (inner_sum-1)
    return entropy

def invert_dict(dictionary):
    '''
    Inverts a dictionary of dictionaries, switching the keys and values.

    :param dictionary: A dictionary with values as dictionaries.

    :return: An inverted dictionary of dictionaries.
    '''
    new_dict = defaultdict(dict)
    for k,v in dictionary.items():
        for k2,v2 in v.items():
            new_dict[k2][k] = v2
    return new_dict

def num_split_localities(partition, graph, locality_col, df):
    '''
    Calculates the number of localities touching 2 or more districts.

    :param partition: The partition to be scored.

    :return: The number of split localities, i.e. the number of localities touching 2 or more districts.
    '''
    dictionary, localities = locality_splits_dict(partition, graph, locality_col, df)
    inverse_dict = invert_dict(dictionary)
    total_splits = 0
    for i in inverse_dict.keys():
        if len(inverse_dict[i]) > 1:
            total_splits += 1
    return total_splits

def pieces_allowed(locality_splits, localities, graph, locality_col, pop_col, to_add=0):
    '''
    Computes the number of pieces allowed, determined by the congressional districts required by the population.

    :param locality_splits: A dictionary with keys as dictrict numbers and values as Counter() dictionaries. These counter dictionaries have pairs County_ID: NUM which counts the number of VTDs in the county in the district. 
    :param localities: A set of the localities in a state.
    :param graph: A graph of the state shapefile. 
    :param locality_col: The string of the locality column's name.
    :param pop_col: The string of the population column's name.
    :param to_add: The allowance to add to the number of pieces allowed for each county.

    :return: The number of pieces allowed per county, i.e. the population of the county divided by the ideal population of a district, rounded to the nearest whole number and summed with the allowance to_add.
    '''
    district_splits ={}
    
    totpop = 0
    for node in graph.nodes:
        totpop += graph.nodes[node][pop_col]
    
    num_districts = len(locality_splits.keys())
    
    for locality in localities:
        sg=graph.subgraph(n for n, v in graph.nodes(data=True) if v[locality_col]==locality)
        pop = 0
        
        for n in sg.nodes():
            pop += sg.node[n][pop_col]
        
        district_splits[locality] = math.ceil(pop/(totpop/num_districts)) + to_add
    return district_splits

def pennsylvania_fouls(partition, graph, locality_col, pop_col, df, to_add=1):
    '''
    Computes the number of Pennsylvania fouls. 
    
    :param partition: The partition to be scored.
    :param graph: A graph of the state shapefile.
    :param locality_col: The string of the locality column's name.
    :param pop_col: The string of the population column's name.
    :param df: A dataframe of the state shapefile.
    :param to_add: The allowance to add to the number of pieces allowed for each county.

    :return: The number of Pennsylvania fouls, i.e. the number of counties which contain more congressional districts than the number required by its population plus one.
    '''
    locality_splits, localities = locality_splits_dict(partition, graph, locality_col, df)
    district_splits = invert_dict(locality_splits)
    pieces = pieces_allowed(locality_splits, localities, graph, locality_col, pop_col, to_add)

    too_many = 0
    for locality in localities:
        if len(district_splits[locality]) > pieces[locality]:
            too_many += 1
    
    return too_many