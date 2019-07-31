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