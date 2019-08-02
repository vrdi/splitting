def locality_splits_dict(partition, locality_col, df):
    """
    From a partition, generates a dictionary of counter dictionaries.

    :param partition: The partition for which a dictionary is being generated.
    :param locality_col: The string of the locality column's name. 
    :param df: The dataframe.

    :return: A dictionary with keys as dictrict numbers and values as Counter() dictionaries. These counter dictionaries have pairs County_ID: NUM which counts the number of VTDs in the county in the district. 
    """
    localitydict = dict(graph.nodes(data=locality_col))
    localities = (set(list(df[locality_col])))

    locality_splits = {k:[] for k in localities}
    locality_splits = {k:[localitydict[v] for v in d] for k,d in partition.assignment.parts.items()}
    locality_splits = {k: Counter(v) for k,v in locality_splits.items()}
    return locality_splits, localities

def num_splits(partition, locality_col, df):
    """
    Calculates the number of counties touching 2 or more districts.

    :param partition: The partition to be scored.
    :param locality_col: The string of the locality column's name. 
    :param df: The dataframe.

    :return: The number of splittings, i.e. the number of counties touching 2 or more districts.
    """
    locality_splits, localities = locality_splits_dict(partition, locality_col, df)

    counter = 0
    for district in locality_splits.keys():
        counter += len(locality_splits[district])
    return counter

def cut_edges_in_district(partition, graph):
    """
    Computes a ratio of the cut edges between two counties over the total number of cut edges.

    :param partition: The partition to be scored.
    :param graph: The graph. 

    :return: The ratio of cut edges between two counties over the total number of cut edges.
    """
    countydict = dict(graph.nodes(data=county_col))

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