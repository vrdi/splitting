# Imports
import os
from gerrychain import Graph, GeographicPartition, Partition, Election, accept
import geopandas as gpd
import numpy as np
import copy
import seaborn as sns
from collections import defaultdict, Counter
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
    :param df: A dataframe of the state shapefile.

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

    :return: The number of cut edges within two counties.
    '''
    locality_dict = dict(graph.nodes(data=locality_col))

    cut_edges_within = 0
    cut_edge_set = partition["cut_edges"]
    for i in cut_edge_set:
        vtd_1 = i[0]
        vtd_2 = i[1]
        county_1 = locality_dict.get(vtd_1)
        county_2 = locality_dict.get(vtd_2)
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

def vtds_to_localities(partition, graph, locality_col, pop_col, localities): #IN PROGRESS
    '''
    Generates a dictionary with keys as districts and values as a dictionaries of locality-population key-values pairs.

    :param partition: The partition for which a dictionary is being generated.
    :param localities: A set of the localities in a state.
    :param graph: A graph of the state shapefile.

    :return: A dictionary with keys as districts and values as dictionaries of locality-population key-value pairs, which represents the population of each locality that is in each district.
    '''
    locality_dict = dict(graph.nodes(data=locality_col))
    district_dict = dict(partition.parts)
    new_district_dict = dict(partition.parts)
    for district in district_dict.keys():
        vtds = district_dict[district]
        locality_pop = {k:0 for k in localities}
        for vtd in vtds:
            locality_pop[locality_dict[vtd]] += graph.nodes[vtd][pop_col]
        new_district_dict[district] = locality_pop
    return new_district_dict

def dictionary_to_score(dictionary): #IN PROGRESS
    '''
    Calculates a symmetric entropy half score.

    :param dictionary: The dictionary to be scored.

    :return: The symmetric entropy half score of the dictionary.
    '''
    district_dict = dictionary
    score = 0
    for district in district_dict.keys():
        localities_and_pops = district_dict[district]
        total = sum(localities_and_pops.values())
        fractional_sum = 0
        for locality in localities_and_pops.keys():
            fractional_sum += np.sqrt(localities_and_pops[locality]/total)
        score += total*fractional_sum
    return score

def symmetric_entropy(partition, graph, locality_col, pop_col, df): #IN PROGRESS
    '''
    Calculates the symmetric entropy score.

    :param partition: The partition to be scored.
    :param graph: A graph of the state shapefile.
    :param locality_col: The string of the locality column's name.
    :param df: A dataframe of the state shapefile.

    :return: The symmetric entropy score.
    '''
    locality_splits, localities = locality_splits_dict(partition, graph, locality_col, df)
    dictionary = vtds_to_localities(partition, graph, locality_col, pop_col, localities)
    return dictionary_to_score(dictionary) + dictionary_to_score(invert_dict(dictionary))

def num_pieces(partition, graph, locality_col):
    '''
    Calculates the number of pieces.

    :param partition: The partition to be scored.
    :param graph: A graph of the state shapefile.
    :param locality_col: The string of the locality column's name.

    :return: Number of pieces, where each piece is formed by cutting the graph by both county and district boundaries.

    '''
    locality_intersections = {}
    for n in graph.nodes():
        locality = graph.nodes[n][locality_col]
        if locality not in locality_intersections:
            locality_intersections[locality] = set([partition.assignment[n]])
        locality_intersections[locality].update([partition.assignment[n]])
    pieces = 0
    for locality in locality_intersections:
        for d in locality_intersections[locality]:
            subgraph = graph.subgraph(
                [x for x in partition.parts[d] if graph.nodes[x][locality_col] == locality]
            )
            pieces += nx.number_connected_components(subgraph)
    return pieces